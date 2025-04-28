require('dotenv').config();

const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const cors = require('cors');
const { spawn } = require('child_process');
const { S3Client, GetObjectCommand, PutObjectCommand } = require('@aws-sdk/client-s3');
const { getSignedUrl } = require('@aws-sdk/s3-request-presigner');
const { generatePresignedUrl, s3Client } = require('./services/s3');
const { promises: fsPromises } = require('fs');
const { v4: uuidv4 } = require('uuid'); // Import uuid for generating job IDs

process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason);
});

process.on('uncaughtException', (error) => {
  console.error('Uncaught Exception:', error);
});

console.log('AWS Config:', {
  region: process.env.AWS_REGION,
  bucket: process.env.AWS_BUCKET_NAME,
  hasAccessKey: !!process.env.AWS_ACCESS_KEY_ID,
  hasSecretKey: !!process.env.AWS_SECRET_ACCESS_KEY
});

const app = express();
const PORT = process.env.PORT;

// In-memory store for job statuses
const jobStore = new Map();
// Set to track currently processing S3 file keys
const processingFiles = new Set();

app.use((req, res, next) => {
  console.log(`${req.method} ${req.path}`, {
    body: req.body,
    query: req.query
  });
  next();
});

app.use(cors({
  origin: true,
  methods: ['GET', 'POST'],
  allowedHeaders: ['Content-Type', 'Accept'],
  credentials: false
}));

app.use(express.json());

function getPythonCommand() {
  const venvPythonPaths = [
    path.join(__dirname, '..', '..', 'venv', 'bin', 'python'),  // mino-study root venv
    path.join(__dirname, '..', 'venv', 'bin', 'python'),        // mino-server venv
    path.join(__dirname, '..', 'bin', 'python'),
    path.join(__dirname, '..', 'env', 'bin', 'python'),
    path.join(__dirname, '..', 'venv', 'Scripts', 'python.exe'),
    path.join(__dirname, '..', 'env', 'Scripts', 'python.exe')
  ];

  for (const venvPath of venvPythonPaths) {
    if (fs.existsSync(venvPath)) {
      console.log('Using Python path:', venvPath);  // Add logging
      return venvPath;
    }
  }
  
  console.log('Falling back to system Python');  // Add logging
  return 'python3'; // fallback to system python
}

// --- Refactored Background Processing Logic ---
async function startBackgroundProcessing(jobId, fileKey) {
  let localPath; // Path for the file downloaded from S3 for processing
  let transcriptFilePath; // Path for the generated transcript file

  console.log(`[Job ${jobId}] Starting background processing for fileKey: ${fileKey}`);

  // This function runs independently after the initial request returns 202
  try {
    // --- Download from S3 for processing ---
    const tempDir = path.join(__dirname, '..', 'temp');
    await fsPromises.mkdir(tempDir, { recursive: true });
    // Unique name for the file downloaded *for this processing job*
    const uniqueProcessingFilename = `${jobId}-${path.basename(fileKey)}`;
    localPath = path.join(tempDir, uniqueProcessingFilename);

    const getCommand = new GetObjectCommand({
      Bucket: process.env.AWS_BUCKET_NAME,
      Key: fileKey
    });
    console.log(`[Job ${jobId}] Fetching from S3 for processing:`, { bucket: process.env.AWS_BUCKET_NAME, key: fileKey });
    const response = await s3Client.send(getCommand);
    const readStream = response.Body;

    if (!readStream) {
      throw new Error('Failed to get video stream from S3 for processing');
    }

    const writeStream = await fsPromises.open(localPath, 'w');
    const chunks = [];
    for await (const chunk of readStream) {
      chunks.push(chunk);
    }
    await writeStream.write(Buffer.concat(chunks));
    await writeStream.close();
    console.log(`[Job ${jobId}] Downloaded file from S3 for processing to: ${localPath}`);

    // --- Spawn Python Script ---
    const pythonCommand = getPythonCommand();
    const pythonProcess = spawn(pythonCommand, [
      path.join(__dirname, 'video_processor.py'),
      localPath,
      localPath // Output base name (Python script handles suffixes)
    ]);

    let pythonOutput = '';
    let pythonError = '';

    pythonProcess.stdout.on('data', (data) => {
      const output = data.toString();
      console.log(`[Job ${jobId}] Python output:`, output);
      pythonOutput += output;
    });

    pythonProcess.stderr.on('data', (data) => {
      const error = data.toString();
      console.log(`[Job ${jobId}] Python error:`, error);
      pythonError += error;
    });

    // --- Handle Python Script Completion ---
    pythonProcess.on('close', async (code) => {
      console.log(`[Job ${jobId}] Python process closed with code: ${code}`);
      let finalResult; // To store parsed result

      try {
        if (code !== 0) {
          throw new Error(`Python process failed with code ${code}: ${pythonError}`);
        }

        const matches = pythonOutput.match(/\{(?:[^{}]|(\{[^{}]*\}))*\}/g);
        if (!matches) {
          throw new Error('No valid JSON found in Python output');
        }
        finalResult = JSON.parse(matches[matches.length - 1]);
        transcriptFilePath = finalResult?.transcript_file; // Store for cleanup

        if (finalResult.status === 'success') {
          let transcriptContent = '';
          if (finalResult.transcript_file) {
            try {
              await fsPromises.access(finalResult.transcript_file, fs.constants.F_OK);
              transcriptContent = await fsPromises.readFile(finalResult.transcript_file, 'utf8');
            } catch (err) {
              console.warn(`[Job ${jobId}] Could not read transcript file (${finalResult.transcript_file}):`, err.message);
              transcriptContent = finalResult.transcript || ''; // Fallback
            }
          } else {
             transcriptContent = finalResult.transcript || ''; // Fallback
          }

          // Generate signed URL for the original video in S3
          const videoCommand = new GetObjectCommand({
            Bucket: process.env.AWS_BUCKET_NAME,
            Key: fileKey
          });
          const originalUrl = await getSignedUrl(s3Client, videoCommand, { expiresIn: 3600 });

          // Update job store on success
          jobStore.set(jobId, {
            status: 'complete',
            data: {
              originalUrl: originalUrl,
              transcript: transcriptContent,
              segments: finalResult.segments || [],
              summary: finalResult.summary,
              keyPoints: finalResult.keyPoints,
              flashcards: finalResult.flashcards,
              // Include stats if available in python output
              stats: finalResult.stats
            },
            endTime: Date.now()
          });
          console.log(`[Job ${jobId}] Completed successfully.`);

        } else {
          throw new Error(finalResult.error || 'Processing failed in Python script');
        }
      } catch (error) {
        console.error(`[Job ${jobId}] Error during Python process close/parsing:`, error);
        // Update job store on error
        jobStore.set(jobId, {
          status: 'error',
          message: error.message || 'An unknown error occurred during processing.',
          details: error.stack,
          endTime: Date.now()
        });
      } finally {
        // --- Release Lock ---
        if (processingFiles.has(fileKey)) {
          processingFiles.delete(fileKey);
          console.log(`[Job ${jobId}] Released lock for fileKey: ${fileKey}`);
        }

        // --- Clean up temporary files for this job ---
        console.log(`[Job ${jobId}] Starting cleanup...`);
        // Cleanup the video file downloaded for processing
        try {
          if (localPath) {
            await fsPromises.access(localPath, fs.constants.F_OK);
            await fsPromises.unlink(localPath);
            console.log(`[Job ${jobId}] Cleaned up temp video file: ${localPath}`);
          }
        } catch (err) {
          if (err.code !== 'ENOENT') {
            console.error(`[Job ${jobId}] Error removing temp video file ${localPath}:`, err);
          } else {
            console.log(`[Job ${jobId}] Temp video file ${localPath} already removed.`);
          }
        }
        // Cleanup the transcript file generated by Python
        try {
          if (transcriptFilePath) {
             await fsPromises.access(transcriptFilePath, fs.constants.F_OK);
             await fsPromises.unlink(transcriptFilePath);
             console.log(`[Job ${jobId}] Cleaned up temp transcript file: ${transcriptFilePath}`);
          }
        } catch (err) {
           if (err.code !== 'ENOENT') {
              console.error(`[Job ${jobId}] Error removing temp transcript file ${transcriptFilePath}:`, err);
           } else {
              console.log(`[Job ${jobId}] Temp transcript file ${transcriptFilePath} already removed.`);
           }
        }
        console.log(`[Job ${jobId}] Cleanup finished.`);
      }
    });

    // --- Handle Python Spawn Error ---
    pythonProcess.on('error', (err) => {
      console.error(`[Job ${jobId}] Failed to start Python process:`, err);
      jobStore.set(jobId, {
        status: 'error',
        message: 'Failed to start processing task.',
        details: err.message,
        endTime: Date.now()
      });
      // Release lock if spawn fails
      if (processingFiles.has(fileKey)) {
        processingFiles.delete(fileKey);
        console.log(`[Job ${jobId}] Released lock for fileKey: ${fileKey} (spawn error)`);
      }
      // Attempt cleanup of downloaded video file
      if (localPath) {
        fsPromises.unlink(localPath).catch(e => console.error(`[Job ${jobId}] Cleanup error after spawn fail:`, e));
      }
    });

  } catch (error) { // Catch errors during the setup phase (e.g., S3 download)
    console.error(`[Job ${jobId}] Error in background processing setup:`, error);
    jobStore.set(jobId, {
      status: 'error',
      message: error.message || 'Failed during setup before processing could start.',
      details: error.stack,
      endTime: Date.now()
    });
    // Release lock on setup error
    if (processingFiles.has(fileKey)) {
      processingFiles.delete(fileKey);
      console.log(`[Job ${jobId}] Released lock for fileKey: ${fileKey} (setup error)`);
    }
     // Attempt cleanup if localPath was determined
     if (localPath) {
       fsPromises.unlink(localPath).catch(e => console.error(`[Job ${jobId}] Cleanup error after setup fail:`, e));
     }
  }
}


// Configure storage
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const uploadDir = path.join(__dirname, '..', 'uploads');
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
    }
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    const uniqueName = `${Date.now()}-${file.originalname}`;
    cb(null, uniqueName);
  }
});

const upload = multer({ storage });

// Upload and process endpoint (Deprecated - Keep for reference or remove later)
// app.post('/upload', ...); // Consider removing or commenting out if not used

app.post('/s3/presigned', async (req, res) => {
  try {
    const { fileName, fileType } = req.body;
    
    if (!fileName || !fileType) {
      return res.status(400).json({ 
        error: 'fileName and fileType are required' 
      });
    }

    const presignedData = await generatePresignedUrl(fileName, fileType);
    res.json(presignedData);
  } catch (error) {
    console.error('Error generating presigned URL:', error);
    res.status(500).json({ 
      error: 'Failed to generate upload URL' 
    });
  }
});

// --- Process S3 Video Endpoint (for direct uploads) ---
app.post('/process/s3-video', async (req, res) => {
  const fileKey = req.body.fileKey;

  if (!fileKey) {
    return res.status(400).json({ error: 'fileKey is required' });
  }

  // --- Lock Check ---
  if (processingFiles.has(fileKey)) {
    console.log(`Processing already in progress for key: ${fileKey}`);
    return res.status(409).json({
      error: 'Processing already in progress for this file. Please wait.'
    });
  }

  // --- Acquire Lock ---
  processingFiles.add(fileKey);
  console.log(`Acquired lock for key: ${fileKey}`);

  // Generate Job ID and store initial state
  const jobId = uuidv4();
  jobStore.set(jobId, { status: 'processing', startTime: Date.now(), fileKey: fileKey });
  console.log(`Job ${jobId} started for fileKey: ${fileKey} (via direct upload)`);

  // Immediately return 202 Accepted
  res.status(202).json({ jobId: jobId });

  // --- Start background processing using the refactored function ---
  startBackgroundProcessing(jobId, fileKey); // No await here

});

// --- Modified YouTube URL Processing Endpoint (Fully Asynchronous) ---
app.post('/process/youtube-url', async (req, res) => {
  const { youtubeUrl } = req.body;
  let tempVideoPath; // Keep track for cleanup in background task
  let fileKey; // Keep track for cleanup/locking in background task

  if (!youtubeUrl) {
    return res.status(400).json({ error: 'youtubeUrl is required' });
  }

  // Basic URL validation
  if (!youtubeUrl.includes('youtube.com') && !youtubeUrl.includes('youtu.be')) {
      return res.status(400).json({ error: 'Invalid YouTube URL provided' });
  }

  // 1. Generate Job ID early
  const jobId = uuidv4();

  // 2. Store Initial Job State ('downloading')
  jobStore.set(jobId, {
    status: 'downloading', // Initial status
    startTime: Date.now(),
    source: 'youtube',
    originalUrl: youtubeUrl
  });
  console.log(`[Job ${jobId}] Registered for YouTube URL: ${youtubeUrl}. Status: downloading.`);

  // 3. Return 202 Accepted Immediately
  res.status(202).json({ jobId: jobId });

  // 4. Perform Tasks in Background
  (async () => {
    try {
      // --- Download using yt-dlp ---
      const tempDir = path.join(__dirname, '..', 'temp');
      await fsPromises.mkdir(tempDir, { recursive: true });
      const uniqueDownloadFilename = `${jobId}-youtube-download.mp4`;
      tempVideoPath = path.join(tempDir, uniqueDownloadFilename);

      console.log(`[Job ${jobId}] Starting yt-dlp download to: ${tempVideoPath}`);
      const ytdlpProcess = spawn('yt-dlp', [
        '-f', 'bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[height<=720]', // Request max 720p MP4
        '-o', tempVideoPath,
        '--', youtubeUrl
      ]);

      let ytdlpError = '';
      ytdlpProcess.stderr.on('data', (data) => {
        console.log(`[Job ${jobId}] yt-dlp stderr: ${data.toString().trim()}`);
        ytdlpError += data.toString();
      });
      ytdlpProcess.stdout.on('data', (data) => {
        console.log(`[Job ${jobId}] yt-dlp stdout: ${data.toString().trim()}`);
      });

      const downloadExitCode = await new Promise((resolve, reject) => {
        ytdlpProcess.on('close', resolve);
        ytdlpProcess.on('error', reject); // Handle spawn errors
      });

      if (downloadExitCode !== 0) {
        throw new Error(`yt-dlp failed with code ${downloadExitCode}. Error: ${ytdlpError}`);
      }
      console.log(`[Job ${jobId}] yt-dlp download successful.`);

      // --- Update Status: Uploading ---
      jobStore.set(jobId, { ...jobStore.get(jobId), status: 'uploading_s3' });
      console.log(`[Job ${jobId}] Status: uploading_s3`);

      // --- Upload to S3 ---
      fileKey = `uploads/youtube-${jobId}-${path.basename(tempVideoPath)}`;
      console.log(`[Job ${jobId}] Uploading ${tempVideoPath} to S3 with key: ${fileKey}`);

      const fileStream = fs.createReadStream(tempVideoPath);
      const uploadCommand = new PutObjectCommand({
          Bucket: process.env.AWS_BUCKET_NAME,
          Key: fileKey,
          Body: fileStream,
          ContentType: 'video/mp4'
      });

      await s3Client.send(uploadCommand);
      console.log(`[Job ${jobId}] S3 upload successful.`);

      // --- Cleanup yt-dlp download (after successful upload) ---
      try {
          await fsPromises.unlink(tempVideoPath);
          console.log(`[Job ${jobId}] Cleaned up temporary yt-dlp download: ${tempVideoPath}`);
          tempVideoPath = null; // Clear path after deletion
      } catch (cleanupErr) {
          console.warn(`[Job ${jobId}] Failed to cleanup yt-dlp download ${tempVideoPath}:`, cleanupErr);
      }

      // --- Acquire Lock (using S3 fileKey) ---
      // Check lock *after* successful S3 upload
      if (processingFiles.has(fileKey)) {
        // This is unlikely now with unique job IDs in fileKey, but good practice
        console.warn(`[Job ${jobId}] Lock for S3 key ${fileKey} already held. Waiting might occur.`);
        // We proceed, startBackgroundProcessing will handle the lock internally
      }
      processingFiles.add(fileKey);
      console.log(`[Job ${jobId}] Acquired lock for S3 key: ${fileKey}`);

      // --- Update Status: Processing ---
      jobStore.set(jobId, { ...jobStore.get(jobId), status: 'processing', fileKey: fileKey });
      console.log(`[Job ${jobId}] Status: processing. FileKey: ${fileKey}`);

      // --- Start Background Processing (Python script phase) ---
      // This function handles the rest: S3 download for Python, spawn, status updates, cleanup, lock release
      startBackgroundProcessing(jobId, fileKey);

    } catch (error) {
      // --- Handle Errors in Background Task ---
      console.error(`[Job ${jobId}] Error during background YouTube processing:`, error);

      // Update job store to 'error'
      jobStore.set(jobId, {
          ...(jobStore.get(jobId) || {}), // Keep existing info if possible
          status: 'error',
          message: `Failed during background processing: ${error.message}`,
          details: error.stack,
          endTime: Date.now(),
          // Ensure these are present even if error happened early
          source: 'youtube',
          originalUrl: youtubeUrl
      });
      console.log(`[Job ${jobId}] Status: error`);

      // Release lock if it was acquired
      if (fileKey && processingFiles.has(fileKey)) {
          processingFiles.delete(fileKey);
          console.log(`[Job ${jobId}] Released lock for key: ${fileKey} due to error.`);
      }

      // Attempt to cleanup temp download file if it still exists
      if (tempVideoPath) {
          fsPromises.unlink(tempVideoPath).catch(cleanupErr => {
              if (cleanupErr.code !== 'ENOENT') {
                  console.warn(`[Job ${jobId}] Error cleaning up temp download ${tempVideoPath} after failure:`, cleanupErr);
              }
          });
      }
    }
  })(); // Immediately invoke the async background function

});


// --- Job Status Endpoint ---
app.get('/process/status/:jobId', (req, res) => {
  const jobId = req.params.jobId;
  const job = jobStore.get(jobId);

  if (!job) {
    return res.status(404).json({ status: 'not_found', message: 'Job not found.' });
  }

  // Optionally remove sensitive details before sending
  const { details, ...jobStatusToSend } = job;

  res.json(jobStatusToSend);
});


// Test endpoint
app.post('/api/transcript/test', (req, res) => {
  const pythonCommand = getPythonCommand();
  const pythonProcess = spawn(pythonCommand, [
    path.join(__dirname, 'video_processor.py'),
    'test'
  ]);


  let outputData = '';

  pythonProcess.stdout.on('data', (data) => {
    outputData += data.toString();
  });

  pythonProcess.stderr.on('data', (data) => {
    console.error(`Python Error: ${data}`);
  });

  pythonProcess.on('close', (code) => {
    if (code !== 0) {
      return res.status(500).json({ success: false, error: 'Failed to process test transcript' });
    }
    try {
      const result = JSON.parse(outputData);
      res.json({ 
        success: true, 
        data: result
      });
    } catch (error) {
      res.status(500).json({ 
        success: false, 
        error: 'Failed to parse Python output',
        rawOutput: outputData 
      });
    }
  });
});

// Serve uploaded videos (If needed for direct access, otherwise remove)
// app.use('/video', express.static(path.join(__dirname, '..', 'uploads')));

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
