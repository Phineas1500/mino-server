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
const OpenAI = require('openai'); // Added OpenAI

// --- Define job data directory ---
const JOB_DATA_DIR = path.join(__dirname, '..', 'jobdata');

// --- Ensure job data directory exists ---
fsPromises.mkdir(JOB_DATA_DIR, { recursive: true })
  .catch(err => console.error('Failed to create job data directory:', err));

// --- Initialize OpenAI Client ---
let openai;
if (process.env.OPENAI_API_KEY) {
  openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
  });
} else {
  console.warn("OPENAI_API_KEY not found in .env file. Chat functionality will be limited.");
}
// --- ---

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

// Helper function to check if yt-dlp is available
async function checkYtDlpAvailable() {
  return new Promise((resolve) => {
    const testProcess = spawn('yt-dlp', ['--version']);
    
    testProcess.on('close', (code) => {
      resolve(code === 0);
    });
    
    testProcess.on('error', (err) => {
      console.log('yt-dlp not found or not executable:', err.message);
      resolve(false);
    });
    
    // Add timeout to prevent hanging
    setTimeout(() => {
      testProcess.kill();
      resolve(false);
    }, 5000);
  });
}

// --- Helper function to read job data ---
async function readJobData(jobId) {
  const filePath = path.join(JOB_DATA_DIR, `${jobId}.json`);
  try {
    const data = await fsPromises.readFile(filePath, 'utf8');
    return JSON.parse(data);
  } catch (error) {
    if (error.code === 'ENOENT') {
      return null; // Job file not found
    }
    console.error(`[Job ${jobId}] Error reading job file:`, error);
    throw error; // Re-throw other errors
  }
}

// --- Helper function to write job data ---
async function writeJobData(jobId, jobData) {
  const filePath = path.join(JOB_DATA_DIR, `${jobId}.json`);
  try {
    await fsPromises.writeFile(filePath, JSON.stringify(jobData, null, 2), 'utf8');
  } catch (error) {
    console.error(`[Job ${jobId}] Error writing job file:`, error);
    throw error;
  }
}

// --- Refactored Background Processing Logic ---
async function startBackgroundProcessing(jobId, fileKey) {
  let localPath; // Path for the file downloaded from S3 for processing
  let transcriptFilePath; // Path for the generated transcript file

  // Helper to update job status consistently (now writes to file)
  const updateJobStatus = async (stage, progress, message = null) => {
    try {
      const currentJob = await readJobData(jobId) || {};
      const updatedJob = {
        ...currentJob,
        status: 'processing', // Keep status as 'processing' until final state
        stage: stage,
        progress: progress,
        ...(message && { message: message }) // Add message if provided
      };
      await writeJobData(jobId, updatedJob);
      console.log(`[Job ${jobId}] Status Update (File): Stage=${stage}, Progress=${progress}%`);
    } catch (error) {
        console.error(`[Job ${jobId}] Failed to update job status file:`, error);
    }
  };

  console.log(`[Job ${jobId}] Starting background processing for fileKey: ${fileKey}`);
  // Initial stage update (now async)
  await updateJobStatus('downloading_from_s3', 25, 'Downloading video for processing...'); // Example: 25%

  // This function runs independently after the initial request returns 202
  try {
    // --- Download from S3 for processing ---
    const tempDir = path.join(__dirname, '..', 'temp');
    await fsPromises.mkdir(tempDir, { recursive: true });
    const uniqueProcessingFilename = `${jobId}-${path.basename(fileKey)}`;
    localPath = path.join(tempDir, uniqueProcessingFilename);

    const getCommand = new GetObjectCommand({
      Bucket: process.env.AWS_BUCKET_NAME,
      Key: fileKey
    });
    console.log(`[Job ${jobId}] Fetching from S3 for processing:`, { bucket: process.env.AWS_BUCKET_NAME, key: fileKey });
    
    let response; // Define response outside try block if needed later
    try {
        response = await s3Client.send(getCommand);
    } catch (s3Error) {
        console.error(`[Job ${jobId}] Failed to send S3 GetObject command:`, s3Error);
        throw new Error(`Failed to initiate download from S3: ${s3Error.message}`); // Re-throw specific error
    }

    const readStream = response.Body;

    // --- Check if readStream is valid ---
    if (!readStream || typeof readStream.pipe !== 'function') {
      console.error(`[Job ${jobId}] Invalid or missing stream from S3 response. Body type: ${typeof readStream}`);
      throw new Error('Failed to get a valid video stream from S3 for processing');
    }
    console.log(`[Job ${jobId}] Received valid stream from S3.`);
    // --- End Check ---


    // --- Stream download directly to file ---
    console.log(`[Job ${jobId}] Starting stream download to ${localPath}...`);
    const writeStream = fs.createWriteStream(localPath);

    console.log(`[Job ${jobId}] Setting up pipe and event listeners...`); // Log before pipe
    await new Promise((resolve, reject) => {
        let readStreamError = null;
        let writeStreamError = null;

        readStream.on('error', (err) => {
            console.error(`[Job ${jobId}] S3 read stream error:`, err);
            readStreamError = err; // Store error
            // Ensure write stream is destroyed to prevent leaks if read fails
            if (!writeStream.destroyed) {
                writeStream.destroy(err);
            }
            reject(err); // Reject the promise
        });

        writeStream.on('error', (err) => {
            console.error(`[Job ${jobId}] File write stream error:`, err);
            writeStreamError = err; // Store error
            // No need to destroy readStream here, S3 SDK handles its cleanup
            reject(err); // Reject the promise
        });

        writeStream.on('finish', () => {
            // Only resolve if no errors occurred on either stream
            if (!readStreamError && !writeStreamError) {
                console.log(`[Job ${jobId}] File write stream finished successfully.`);
                resolve();
            } else {
                console.log(`[Job ${jobId}] File write stream finished, but an error occurred earlier.`);
                // Reject with the first error encountered if finish event still fires after error
                reject(writeStreamError || readStreamError || new Error("Stream finished with pending errors"));
            }
        });
        
        // Initiate the pipe *after* listeners are attached
        console.log(`[Job ${jobId}] Initiating pipe...`);
        readStream.pipe(writeStream);
    });
    // writeStream is automatically closed on 'finish' or 'error' when using pipe
    console.log(`[Job ${jobId}] Downloaded file from S3 for processing to: ${localPath}`);
    // --- End stream download ---

    await updateJobStatus('preparing_script', 30, 'Preparing analysis script...'); // Example: 30%

    // --- Spawn Python Script ---
    const pythonCommand = getPythonCommand();
    await updateJobStatus('processing_video', 35, 'Starting video analysis...'); // Example: 35%
    const pythonProcess = spawn(pythonCommand, [
      path.join(__dirname, 'video_processor.py'), // Ensure this path is correct
      localPath,
      localPath // Output base name (Python script handles suffixes)
    ]);

    let pythonOutput = '';
    let pythonError = '';
    let stdoutBuffer = ''; // <--- Add a buffer for stdout

    pythonProcess.stdout.on('data', (data) => {
      stdoutBuffer += data.toString(); // Append new data to buffer
      let newlineIndex;

      // Process all complete lines in the buffer
      while ((newlineIndex = stdoutBuffer.indexOf('\n')) >= 0) {
        const line = stdoutBuffer.substring(0, newlineIndex).trim(); // Get the complete line
        stdoutBuffer = stdoutBuffer.substring(newlineIndex + 1); // Remove the processed line from buffer

        if (line) { // Process non-empty lines
          console.log(`[Job ${jobId}] Python stdout line:`, line); // Log the processed line

          const progressMatch = line.match(/PROGRESS:\s*(\d+)\s+STAGE:\s*(\S+)(?:\s+MESSAGE:\s*(.*))?$/);
          if (progressMatch) {
              const pyProgress = parseInt(progressMatch[1], 10);
              const pyStage = progressMatch[2];
              const pyMessage = progressMatch[3] || `Processing stage: ${pyStage}`;

              const overallProgress = Math.max(35, Math.min(95, pyProgress)); // Clamp progress
              updateJobStatus(pyStage, overallProgress, pyMessage); // Call async update
          } else {
               // Assume lines not matching PROGRESS are part of the final JSON or other logs
               // Append only non-progress lines intended for final JSON parsing
               if (!line.startsWith('PROGRESS:')) {
                   pythonOutput += line + '\n';
               }
          }
        }
      }
      // Any remaining data in stdoutBuffer is an incomplete line, wait for more data
    });

    pythonProcess.stderr.on('data', (data) => {
      // ... (stderr handling remains the same) ...
      const error = data.toString();
      console.log(`[Job ${jobId}] Python error:`, error);
      pythonError += error;
    });

    pythonProcess.on('close', async (code) => {
      // --- Process any remaining data in the buffer ---
      if (stdoutBuffer.trim()) {
          console.log(`[Job ${jobId}] Python stdout (remaining buffer):`, stdoutBuffer.trim());
          // Decide if remaining buffer should be part of pythonOutput for JSON parsing
          if (!stdoutBuffer.trim().startsWith('PROGRESS:')) {
              pythonOutput += stdoutBuffer.trim() + '\n';
          }
          stdoutBuffer = ''; // Clear buffer
      }
      // --- End processing remaining buffer ---

      console.log(`[Job ${jobId}] Python process closed with code: ${code}`);
      await updateJobStatus('parsing_results', 95, 'Parsing analysis results...'); // Example: 95%
      let finalResult; // To store parsed result

      try {
        if (code !== 0) {
          throw new Error(`Python process failed with code ${code}: ${pythonError}`);
        }

        // --- Safely find the last JSON object ---
        const jsonBlobs = pythonOutput.match(/\{(?:[^{}]|(\{(?:[^{}]|(\{[^{}]*\}))*\}))*\}/g);
        if (!jsonBlobs || jsonBlobs.length === 0) {
            // Look for specific error markers if no JSON found
            if (pythonOutput.includes("Traceback") || pythonError) {
                 throw new Error(`Python script error: ${pythonError || pythonOutput.slice(-500)}`);
            }
            throw new Error('No valid JSON output found from Python script.');
        }
        try {
            finalResult = JSON.parse(jsonBlobs[jsonBlobs.length - 1]);
        } catch (parseError) {
            console.error(`[Job ${jobId}] Failed to parse final JSON: ${parseError}. Raw: ${jsonBlobs[jsonBlobs.length - 1]}`);
            throw new Error(`Failed to parse results from analysis script. ${parseError.message}`);
        }
        // --- End Safe JSON Parsing ---

        transcriptFilePath = finalResult?.transcript_file; // Store for cleanup

        if (finalResult.status === 'success') {
          await updateJobStatus('finalizing', 98, 'Finalizing results...'); // Example: 98%
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

          // Update job store on success (now writes to file)
          const successData = {
            status: 'complete',
            stage: 'complete',
            progress: 100,
            data: {
              fileKey: fileKey,
              transcript: transcriptContent,
              segments: finalResult.segments || [],
              summary: finalResult.summary,
              keyPoints: finalResult.keyPoints,
              flashcards: finalResult.flashcards,
              stats: finalResult.stats
            },
            endTime: Date.now(),
            // --- Ensure source and originalUrl are preserved if they exist ---
            ...(await readJobData(jobId) || {}).source && { source: (await readJobData(jobId)).source },
            ...(await readJobData(jobId) || {}).originalUrl && { originalUrl: (await readJobData(jobId)).originalUrl },
            // --- ---
          };
          await writeJobData(jobId, successData);
          console.log(`[Job ${jobId}] Completed successfully (File).`);

        } else {
          throw new Error(finalResult.error || 'Processing failed in Python script');
        }
      } catch (error) {
        console.error(`[Job ${jobId}] Error during Python process close/parsing:`, error);
        // Update job store on error (now writes to file)
        const currentJob = await readJobData(jobId) || {};
        const errorData = {
            ...currentJob, // Preserve existing data like startTime, fileKey, source etc.
            status: 'error',
            stage: 'error',
            progress: currentJob.progress || 0, // Keep progress where it failed
            message: error.message || 'An unknown error occurred during processing.',
            details: error.stack,
            endTime: Date.now()
        };
        await writeJobData(jobId, errorData);
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
    pythonProcess.on('error', async (err) => { // Make handler async
      console.error(`[Job ${jobId}] Failed to start Python process:`, err);
      const currentJob = await readJobData(jobId) || {};
      const spawnErrorData = {
        ...currentJob,
        status: 'error',
        stage: 'error',
        progress: currentJob.progress || 30, // Progress where it failed
        message: 'Failed to start processing task.',
        details: err.message,
        endTime: Date.now()
      };
      await writeJobData(jobId, spawnErrorData);
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
    console.error(`[Job ${jobId}] Error in background processing setup (including download):`, error); // Updated log message
    const currentJob = await readJobData(jobId) || {}; // Read existing data if available
    const setupErrorData = {
        ...currentJob, // Preserve existing data
        status: 'error',
        stage: 'error',
        progress: currentJob.progress || 0, // Progress where it failed
        message: error.message || 'Failed during setup before processing could start.',
        details: error.stack,
        endTime: Date.now()
    };
    await writeJobData(jobId, setupErrorData); // Write error status to file
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

  // Generate Job ID and store initial state (now writes to file)
  const jobId = uuidv4();
  const initialJobData = {
      status: 'processing',
      stage: 'pending',
      progress: 5,
      message: 'Processing request received...',
      startTime: Date.now(),
      fileKey: fileKey,
      source: 's3-upload'
  };
  try {
      await writeJobData(jobId, initialJobData); // Write initial state
      console.log(`Job ${jobId} started (File) for fileKey: ${fileKey} (via direct upload)`);

      // Immediately return 202 Accepted
      res.status(202).json({ jobId: jobId });

      // --- Start background processing ---
      startBackgroundProcessing(jobId, fileKey); // No await here

  } catch (error) {
      console.error(`[Job ${jobId}] Failed to write initial job data:`, error);
      // Release lock if initial write fails
      if (processingFiles.has(fileKey)) {
          processingFiles.delete(fileKey);
          console.log(`[Job ${jobId}] Released lock for key: ${fileKey} (initial write error)`);
      }
      res.status(500).json({ error: 'Failed to initialize processing job.' });
  }
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

  // Helper to update job status consistently (now writes to file)
  const updateYoutubeJobStatus = async (stage, progress, message = null, extraData = {}) => {
    try {
        const currentJob = await readJobData(jobId) || {};
        const updatedJob = {
            ...currentJob,
            status: 'processing',
            stage: stage,
            progress: progress,
            ...(message && { message: message }),
            ...extraData
        };
        await writeJobData(jobId, updatedJob);
        console.log(`[Job ${jobId}] Status Update (File): Stage=${stage}, Progress=${progress}%`);
    } catch (error) {
        console.error(`[Job ${jobId}] Failed to update YouTube job status file:`, error);
    }
  };

  // 2. Store Initial Job State ('pending') (now writes to file)
  const initialYoutubeJobData = {
    status: 'processing',
    stage: 'pending',
    progress: 5,
    message: 'YouTube processing request received...',
    startTime: Date.now(),
    source: 'youtube',
    originalUrl: youtubeUrl
  };

  try {
      await writeJobData(jobId, initialYoutubeJobData); // Write initial state
      console.log(`[Job ${jobId}] Registered (File) for YouTube URL: ${youtubeUrl}. Status: pending.`);

      // 3. Return 202 Accepted Immediately
      res.status(202).json({ jobId: jobId });

      // 4. Perform Tasks in Background
      (async () => {
        try {
          // --- Check yt-dlp availability first ---
          const ytDlpAvailable = await checkYtDlpAvailable();
          if (!ytDlpAvailable) {
            throw new Error('yt-dlp is not installed or not accessible. Please install yt-dlp to download YouTube videos. Run: pip install yt-dlp');
          }

          // --- Download using yt-dlp ---
          await updateYoutubeJobStatus('downloading_youtube', 10, 'Downloading video from YouTube...');
          const tempDir = path.join(__dirname, '..', 'temp');
          await fsPromises.mkdir(tempDir, { recursive: true });
          const uniqueDownloadFilename = `${jobId}-youtube-download.mp4`;
          tempVideoPath = path.join(tempDir, uniqueDownloadFilename);

          // Function to attempt download with different yt-dlp configurations
          const attemptDownload = async (attemptNumber, config) => {
            console.log(`[Job ${jobId}] Download attempt ${attemptNumber} with config: ${config.name}`);
            
            return new Promise((resolve, reject) => {
              const ytdlpProcess = spawn('yt-dlp', config.args);
              let ytdlpError = '';

              ytdlpProcess.stderr.on('data', (data) => {
                const stderrText = data.toString();
                console.log(`[Job ${jobId}] yt-dlp stderr (attempt ${attemptNumber}): ${stderrText.trim()}`);
                ytdlpError += stderrText;
                
                // Parse progress for UI updates
                const progressMatch = stderrText.match(/\[download\]\s+(\d+(\.\d+)?%)/);
                if (progressMatch && progressMatch[1]) {
                  const ytProgress = parseFloat(progressMatch[1]);
                  const overallProgress = 10 + Math.round(ytProgress * 0.10);
                  updateYoutubeJobStatus('downloading_youtube', overallProgress, `Downloading from YouTube (${ytProgress.toFixed(1)}%) - Attempt ${attemptNumber}...`);
                }
              });

              ytdlpProcess.stdout.on('data', (data) => {
                console.log(`[Job ${jobId}] yt-dlp stdout (attempt ${attemptNumber}): ${data.toString().trim()}`);
              });

              ytdlpProcess.on('close', (code) => {
                if (code === 0) {
                  resolve({ success: true, attempt: attemptNumber });
                } else {
                  resolve({ success: false, code, error: ytdlpError, attempt: attemptNumber });
                }
              });

              ytdlpProcess.on('error', (err) => {
                reject({ success: false, error: err.message, attempt: attemptNumber });
              });
            });
          };

          // Define different download configurations with fallbacks
          const downloadConfigs = [
            {
              name: 'basic_audio_video',
              args: [
                '--format', 'bestaudio+bestvideo/best',  // Try to merge audio+video or get best with audio
                '--merge-output-format', 'mp4',
                '--no-warnings',
                '-o', tempVideoPath,
                '--', youtubeUrl
              ]
            },
            {
              name: 'optimized',
              args: [
                '-f', 'best[height<=720][acodec!=none]/best[height<=1080][acodec!=none]/best[acodec!=none]',  // Ensure audio is present
                '--merge-output-format', 'mp4',
                '--prefer-ffmpeg',
                '--postprocessor-args', 'ffmpeg:-c:v libx264 -c:a aac -movflags +faststart',
                '--no-warnings',
                '--ignore-errors',
                '--extract-flat', 'false',
                '--no-playlist',
                '-o', tempVideoPath,
                '--progress',
                '--', youtubeUrl
              ]
            },
            {
              name: 'simple',
              args: [
                '-f', 'best[acodec!=none]/worst[acodec!=none]',  // Ensure audio is present
                '--merge-output-format', 'mp4',
                '--no-warnings',
                '-o', tempVideoPath,
                '--progress',
                '--', youtubeUrl
              ]
            }
          ];

          console.log(`[Job ${jobId}] Starting yt-dlp download to: ${tempVideoPath}`);
          let downloadResult = null;
          let lastError = '';

          // Try each configuration
          for (let i = 0; i < downloadConfigs.length; i++) {
            try {
              downloadResult = await attemptDownload(i + 1, downloadConfigs[i]);
              
              if (downloadResult.success) {
                console.log(`[Job ${jobId}] yt-dlp download successful on attempt ${downloadResult.attempt}`);
                break;
              } else {
                lastError = downloadResult.error;
                console.log(`[Job ${jobId}] Attempt ${downloadResult.attempt} failed with code ${downloadResult.code}`);
                
                // If this isn't the last attempt, wait a bit before retrying
                if (i < downloadConfigs.length - 1) {
                  console.log(`[Job ${jobId}] Waiting 2 seconds before next attempt...`);
                  await new Promise(resolve => setTimeout(resolve, 2000));
                }
              }
            } catch (error) {
              console.log(`[Job ${jobId}] Attempt ${i + 1} threw error:`, error);
              lastError = error.error || error.message;
            }
          }

          // Check if all attempts failed
          if (!downloadResult || !downloadResult.success) {
            const errorMessage = `Failed to download YouTube video after ${downloadConfigs.length} attempts. ` +
              `This may be due to: 1) YouTube's anti-bot measures, 2) The video being private/restricted, ` +
              `3) An outdated yt-dlp version, or 4) Network issues. ` +
              `Last error: ${lastError}. ` +
              `Consider updating yt-dlp: pip install --upgrade yt-dlp`;
            throw new Error(errorMessage);
          }
          await updateYoutubeJobStatus('uploading_to_s3', 20, 'Uploading video to storage...'); // Example: 20%

          // --- Upload to S3 ---
          fileKey = `uploads/youtube-${path.basename(tempVideoPath)}`;
          console.log(`[Job ${jobId}] Uploading ${tempVideoPath} to S3 with key: ${fileKey}`);

          const fileStream = fs.createReadStream(tempVideoPath);
          const uploadCommand = new PutObjectCommand({
              Bucket: process.env.AWS_BUCKET_NAME,
              Key: fileKey,
              Body: fileStream,
              ContentType: 'video/mp4'
              // Add progress tracking here if needed using Upload from @aws-sdk/lib-storage
          });

          await s3Client.send(uploadCommand);
          console.log(`[Job ${jobId}] S3 upload successful.`);
          await updateYoutubeJobStatus('queued_for_processing', 22, 'Video queued for analysis...', { fileKey: fileKey }); // Example: 22%

          // --- Cleanup yt-dlp download (after successful upload) ---
          try {
              await fsPromises.unlink(tempVideoPath);
              console.log(`[Job ${jobId}] Cleaned up temporary yt-dlp download: ${tempVideoPath}`);
              tempVideoPath = null; // Clear path after deletion
          } catch (cleanupErr) {
              console.warn(`[Job ${jobId}] Failed to cleanup yt-dlp download ${tempVideoPath}:`, cleanupErr);
          }

          // --- Acquire Lock (using S3 fileKey) ---
          if (processingFiles.has(fileKey)) {
            console.warn(`[Job ${jobId}] Lock for S3 key ${fileKey} already held. Waiting might occur.`);
          }
          processingFiles.add(fileKey);
          console.log(`[Job ${jobId}] Acquired lock for S3 key: ${fileKey}`);

          // --- Update Status before handing off ---
          // The next stage ('downloading_from_s3') will be set by startBackgroundProcessing
          await updateYoutubeJobStatus('initiating_processing', 24, 'Initiating video analysis...', { fileKey: fileKey }); // Example: 24%

          // --- Start Background Processing (Python script phase) ---
          startBackgroundProcessing(jobId, fileKey); // This function will handle subsequent stages

        } catch (error) {
          // --- Handle Errors in Background Task ---
          console.error(`[Job ${jobId}] Error during background YouTube processing:`, error);
          const currentJob = await readJobData(jobId) || {}; // Read existing data
          const youtubeErrorData = {
              ...currentJob, // Preserve existing info
              status: 'error',
              stage: 'error',
              progress: currentJob.progress || 0,
              message: `Failed during background processing: ${error.message}`,
              details: error.stack,
              endTime: Date.now(),
              // Ensure source and originalUrl are present
              source: 'youtube',
              originalUrl: youtubeUrl
          };
          await writeJobData(jobId, youtubeErrorData); // Write error status to file
          console.log(`[Job ${jobId}] Status: error (File)`);
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

  } catch (error) {
      console.error(`[Job ${jobId}] Failed to write initial YouTube job data:`, error);
      res.status(500).json({ error: 'Failed to initialize YouTube processing job.' });
  }
});

// --- Job Status Endpoint ---
app.get('/process/status/:jobId', async (req, res) => {
  const jobId = req.params.jobId;
  let job; // Use let instead of const

  try {
      job = await readJobData(jobId); // Read from file
  } catch (error) {
      // Error already logged in readJobData
      return res.status(500).json({ status: 'error', message: 'Failed to read job status.' });
  }

  if (!job) {
    return res.status(404).json({ status: 'not_found', message: 'Job not found.' });
  }

  // --- Generate signed URL on demand for completed jobs ---
  if (job.status === 'complete' && job.data && job.data.fileKey) {
    try {
      const command = new GetObjectCommand({
        Bucket: process.env.AWS_BUCKET_NAME,
        Key: job.data.fileKey
      });
      // Generate a fresh URL valid for 1 hour (or adjust as needed)
      const signedUrl = await getSignedUrl(s3Client, command, { expiresIn: 3600 });
      
      // Add the fresh URL to the data being sent
      job.data.originalUrl = signedUrl; 
      
    } catch (error) {
      console.error(`[Job ${jobId}] Error generating signed URL for status request:`, error);
      // Optionally handle the error, e.g., return status without URL or an error status
      // For now, we'll proceed without the URL if generation fails
      job.data.originalUrl = null; // Indicate URL generation failed
    }
  }
  // --- End URL generation ---

  // Optionally remove sensitive details before sending
  const { details, ...jobStatusToSend } = job;

  res.json(jobStatusToSend);
});

// --- Chat API Endpoint ---
app.post('/api/chat', async (req, res) => {
  const { message, jobId } = req.body;

  if (!openai) {
    return res.status(503).json({ message: "Chat service is not configured. Missing API key." });
  }

  if (!message || !jobId) {
    return res.status(400).json({ message: 'Message and jobId are required.' });
  }

  console.log(`[Chat API] Received message for jobId ${jobId}: "${message}"`);

  try {
    const jobData = await readJobData(jobId);

    if (!jobData) {
      return res.status(404).json({ message: `Job data not found for ID: ${jobId}. Cannot provide context for chat.` });
    }

    if (jobData.status !== 'complete' || !jobData.data) {
      return res.status(400).json({ message: `Job ${jobId} is not yet complete or has no data. Please wait for processing to finish.` });
    }

    const { summary, keyPoints, transcript } = jobData.data;
    
    let limitedContext = "";
    if (summary) {
      limitedContext += `Video Summary:\n${summary}\n\n`;
    }
    if (keyPoints && keyPoints.length > 0) {
      limitedContext += `Key Points:\n${keyPoints.join('\n')}\n\n`;
    }

    // First call to the LLM
    const initialCompletion = await openai.chat.completions.create({
      model: "gpt-4.1-nano-2025-04-14", 
      messages: [
        {
          role: "system",
          content: `You are a helpful assistant discussing a video. Provided context includes a summary and key points.\nVideo Context:\n${limitedContext}\nYour task is to answer the user's question based *only* on the Video Context provided above. If you can provide a confident and accurate answer using *only* this summary and key points, please do so directly. If, and only if, you determine that the full video transcript is *absolutely necessary* to answer the question accurately, then respond with the exact phrase:\nNEED_FULL_TRANSCRIPT\nDo not add any other text or explanation if you output NEED_FULL_TRANSCRIPT. Otherwise, answer the question.`
        },
        {
          role: "user",
          content: message
        }
      ],
      temperature: 0.7,
      max_tokens: 300, 
    });

    let botReply = initialCompletion.choices[0]?.message?.content?.trim();

    if (botReply === "NEED_FULL_TRANSCRIPT") {
      console.log(`[Chat API] LLM indicated NEED_FULL_TRANSCRIPT for jobId ${jobId}. Making a second call with full transcript.`);
      
      let fullContext = limitedContext; // Start with summary and keypoints
      if (transcript) {
        fullContext += `Full Transcript:\n${transcript}\n\n`; // Append full transcript
      }

      const secondCompletion = await openai.chat.completions.create({
        model: "gpt-4.1-nano-2025-04-14", 
        messages: [
          {
            role: "system",
            content: `You are a helpful assistant discussing a video. You previously indicated that the full transcript was needed to answer the user's question. The full context including summary, key points, and the transcript is now provided.\nFull Video Context:\n${fullContext}\nPlease now answer the user's original question using this complete context.`
          },
          {
            role: "user",
            content: message // User's original message
          }
        ],
        temperature: 0.7,
        max_tokens: 300, // Adjust as needed, ensure it's enough for a detailed answer
      });
      botReply = secondCompletion.choices[0]?.message?.content?.trim() || "Sorry, I couldn't generate a response after fetching the full transcript.";
    }
    
    if (!botReply) {
        botReply = "Sorry, I couldn't generate a response.";
    }

    res.json({ reply: botReply });

  } catch (error) {
    console.error(`[Chat API] Error processing chat for jobId ${jobId}:`, error);
    if (error.response) { 
        console.error('[Chat API] OpenAI Error Status:', error.response.status);
        console.error('[Chat API] OpenAI Error Data:', error.response.data);
    }
    res.status(500).json({ message: 'An error occurred while processing your chat message.' });
  }
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
