// Modify the top imports in server.js
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

// Add the error handling right after imports
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

// Add a set to track currently processing file keys
const processingFiles = new Set();

// Add processMap to track running Python processes (Optional, can be removed if cleanup is handled differently)
// const processMap = new Map(); // We might not need this if we rely on jobStore

// Add cleanup function (Optional, can be removed if cleanup is handled differently)
// function cleanupProcess(processId) { ... }

// --- Modified /process/s3-video Endpoint ---
app.post('/process/s3-video', async (req, res) => {
  let localPath; // To be used in the 'close' handler for cleanup
  let transcriptFilePath; // To store the path for cleanup
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
  console.log(`Job ${jobId} started for fileKey: ${fileKey}`);

  // Immediately return 202 Accepted
  res.status(202).json({ jobId: jobId });

  // --- Start background processing ---
  // Use a self-executing async function to handle the background task
  // This avoids blocking the main request handler
  (async () => {
    try {
      console.log(`[Job ${jobId}] Processing video from S3: ${fileKey}`);

      const tempDir = path.join(__dirname, '..', 'temp');
      await fsPromises.mkdir(tempDir, { recursive: true });
      const uniqueTempFilename = `${Date.now()}-${path.basename(fileKey)}`;
      localPath = path.join(tempDir, uniqueTempFilename);

      const getCommand = new GetObjectCommand({
        Bucket: process.env.AWS_BUCKET_NAME,
        Key: fileKey
      });
      console.log(`[Job ${jobId}] Fetching from S3:`, { bucket: process.env.AWS_BUCKET_NAME, key: fileKey });
      const response = await s3Client.send(getCommand);
      const readStream = response.Body;

      if (!readStream) {
        throw new Error('Failed to get video stream from S3');
      }

      const writeStream = await fsPromises.open(localPath, 'w');
      const chunks = [];
      for await (const chunk of readStream) {
        chunks.push(chunk);
      }
      await writeStream.write(Buffer.concat(chunks));
      await writeStream.close();
      console.log(`[Job ${jobId}] Downloaded file from S3 to: ${localPath}`);

      const pythonCommand = getPythonCommand();
      const pythonProcess = spawn(pythonCommand, [
        path.join(__dirname, 'video_processor.py'),
        localPath,
        localPath // Output base name
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

      pythonProcess.on('close', async (code) => {
        console.log(`[Job ${jobId}] Python process closed with code: ${code}`);
        let finalResult; // To store parsed result for cleanup

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
                flashcards: finalResult.flashcards
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
            details: error.stack, // Optional: include stack trace
            endTime: Date.now()
          });
        } finally {
          // --- Release Lock ---
          if (processingFiles.has(fileKey)) {
            processingFiles.delete(fileKey);
            console.log(`[Job ${jobId}] Released lock for key: ${fileKey}`);
          }

          // --- Clean up temporary files ---
          console.log(`[Job ${jobId}] Starting cleanup...`);
          try {
            if (localPath) {
              await fsPromises.access(localPath, fs.constants.F_OK);
              await fsPromises.unlink(localPath);
              console.log(`[Job ${jobId}] Cleaned up temp video file: ${localPath}`);
            } else {
               console.log(`[Job ${jobId}] No local video path to clean.`);
            }
          } catch (err) {
            if (err.code !== 'ENOENT') {
              console.error(`[Job ${jobId}] Error removing temp video file ${localPath}:`, err);
            } else {
              console.log(`[Job ${jobId}] Temp video file ${localPath} already removed or never existed.`);
            }
          }
          try {
            if (transcriptFilePath) {
               await fsPromises.access(transcriptFilePath, fs.constants.F_OK);
               await fsPromises.unlink(transcriptFilePath);
               console.log(`[Job ${jobId}] Cleaned up temp transcript file: ${transcriptFilePath}`);
            } else {
               console.log(`[Job ${jobId}] No transcript file path to clean.`);
            }
          } catch (err) {
             if (err.code !== 'ENOENT') {
                console.error(`[Job ${jobId}] Error removing temp transcript file ${transcriptFilePath}:`, err);
             } else {
                console.log(`[Job ${jobId}] Temp transcript file ${transcriptFilePath} already removed or never existed.`);
             }
          }
          console.log(`[Job ${jobId}] Cleanup finished.`);
        }
      });

      pythonProcess.on('error', (err) => {
        console.error(`[Job ${jobId}] Failed to start Python process:`, err);
        // Update job store on spawn error
        jobStore.set(jobId, {
          status: 'error',
          message: 'Failed to start processing task.',
          details: err.message,
          endTime: Date.now()
        });
        // Release lock if spawn fails
        if (processingFiles.has(fileKey)) {
          processingFiles.delete(fileKey);
          console.log(`[Job ${jobId}] Released lock for key: ${fileKey} (spawn error)`);
        }
        // Attempt cleanup even on spawn error
        if (localPath) {
          fsPromises.unlink(localPath).catch(e => console.error(`[Job ${jobId}] Cleanup error after spawn fail:`, e));
        }
      });

    } catch (error) {
      console.error(`[Job ${jobId}] Error in background processing setup:`, error);
      // Update job store on setup error (e.g., S3 download failed)
      jobStore.set(jobId, {
        status: 'error',
        message: error.message || 'Failed during setup before processing could start.',
        details: error.stack,
        endTime: Date.now()
      });
      // Release lock on setup error
      if (processingFiles.has(fileKey)) {
        processingFiles.delete(fileKey);
        console.log(`[Job ${jobId}] Released lock for key: ${fileKey} (setup error)`);
      }
       // Attempt cleanup even on setup error
       if (localPath) {
         fsPromises.unlink(localPath).catch(e => console.error(`[Job ${jobId}] Cleanup error after setup fail:`, e));
       }
    }
  })(); // Immediately invoke the async function

});

// --- New GET /process/status/:jobId Endpoint ---
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
