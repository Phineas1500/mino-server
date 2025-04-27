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

// Upload and process endpoint
app.post('/upload', upload.single('video'), async (req, res) => {
  try {
    console.log('=== Upload Request Started ===');
    console.log('Headers:', req.headers);
    console.log('Body:', req.body);

    if (!req.file) {
      console.log('No file received in request');
      return res.status(400).json({ error: 'No file uploaded' });
    }

    console.log('File details:', {
      originalname: req.file.originalname,
      mimetype: req.file.mimetype,
      size: req.file.size,
      path: req.file.path
    });
    
    const inputPath = req.file.path;
    const outputPath = inputPath;

    console.log('Starting transcription process...');
    console.log('Input path:', inputPath);
    
    const pythonCommand = getPythonCommand();
    const pythonProcess = spawn(pythonCommand, [
      path.join(__dirname, 'video_processor.py'),
      inputPath,
      outputPath
    ]);
    

    let pythonOutput = '';
    let pythonError = '';

    pythonProcess.stdout.on('data', (data) => {
      const output = data.toString();
      console.log('Raw Python output:', output); // Add this line
      pythonOutput += output;
    });

    pythonProcess.stderr.on('data', (data) => {
      const error = data.toString();
      console.log('Python error:', error);
      pythonError += error;
    });

    pythonProcess.on('close', (code) => {
      if (code !== 0) {
        console.error('Python process failed with code:', code);
        return res.status(500).json({ 
          success: false, 
          error: 'Failed to process video',
          pythonError 
        });
      }

      try {
        // Find the last complete JSON object in the output
        const matches = pythonOutput.match(/\{(?:[^{}]|(\{[^{}]*\}))*\}/g);
        if (!matches) {
          throw new Error('No valid JSON found in Python output');
        }
        // Take the last complete JSON object
        const lastJson = matches[matches.length - 1];
        const result = JSON.parse(lastJson);
        
        if (result.status === 'success') {
          // Read the transcript file
          const transcriptContent = fs.readFileSync(result.transcript_file, 'utf8');
          
          res.json({
            success: true,
            url: `http://localhost:${PORT}/video/${path.basename(inputPath)}`,
            data: {
              transcript: transcriptContent,
              segments: result.segments,
              summary: result.summary,
              keyPoints: result.keyPoints,
              flashcards: result.flashcards
            }
          });
        } else {
          res.status(500).json({ 
            success: false, 
            error: result.error || 'Processing failed' 
          });
        }
      } catch (error) {
        console.error('Error parsing Python output:', error);
        console.error('Raw Python output received:', pythonOutput); // Add this line
        res.status(500).json({ 
          success: false, 
          error: 'Failed to parse Python output',
          pythonOutput
        });
      }
    });
  } catch (error) {
    console.error('Server error:', error);
    res.status(500).json({ 
      success: false, 
      error: error.message 
    });
  }
});

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

// Add processMap to track running Python processes
const processMap = new Map();

// Add cleanup function
function cleanupProcess(processId) {
  const process = processMap.get(processId);
  if (process) {
    process.kill();
    processMap.delete(processId);
    console.log(`Cleaned up process ${processId}`);
  }
}

// Replace the existing /process/s3-video endpoint with this updated version:
app.post('/process/s3-video', async (req, res) => {
  let localPath;
  let result;
  const processId = Date.now().toString();
  let fileKey; // Define fileKey here to access in finally block

  // Set up cleanup on client disconnect
  req.on('close', () => {
    if (processMap.has(processId)) {
      console.log('Client disconnected, cleaning up...');
      cleanupProcess(processId);
      // Also remove from processingFiles if the request is aborted mid-process
      if (fileKey && processingFiles.has(fileKey)) {
        processingFiles.delete(fileKey);
        console.log(`Removed lock for ${fileKey} due to client disconnect`);
      }
    }
  });

  try {
    // Extract fileKey earlier
    fileKey = req.body.fileKey;

    if (!fileKey) {
      return res.status(400).json({
        error: 'fileKey is required'
      });
    }

    // --- Lock Check ---
    if (processingFiles.has(fileKey)) {
      console.log(`Processing already in progress for key: ${fileKey}`);
      // Send a 409 Conflict status code
      return res.status(409).json({
        error: 'Processing already in progress for this file. Please wait.'
      });
    }
    // --- Acquire Lock ---
    processingFiles.add(fileKey);
    console.log(`Acquired lock for key: ${fileKey}`);


    console.log('Processing video from S3:', fileKey);

    // Create temp directory if it doesn't exist
    const tempDir = path.join(__dirname, '..', 'temp');
    await fsPromises.mkdir(tempDir, { recursive: true });

    // Use a unique name for the local temp file to avoid conflicts
    const uniqueTempFilename = `${Date.now()}-${path.basename(fileKey)}`;
    localPath = path.join(tempDir, uniqueTempFilename);

    // Download original video from S3
    const getCommand = new GetObjectCommand({
      Bucket: process.env.AWS_BUCKET_NAME,
      Key: fileKey
    });

    console.log('Fetching from S3:', {
      bucket: process.env.AWS_BUCKET_NAME,
      key: fileKey
    });

    const response = await s3Client.send(getCommand);
    const readStream = response.Body;

    if (!readStream) {
      // Release lock before throwing error
      processingFiles.delete(fileKey);
      console.log(`Released lock for key: ${fileKey} (S3 stream error)`);
      throw new Error('Failed to get video stream from S3');
    }

    // Write the original video locally
    const writeStream = await fsPromises.open(localPath, 'w');
    const chunks = [];
    for await (const chunk of readStream) {
      chunks.push(chunk);
    }
    await writeStream.write(Buffer.concat(chunks));
    await writeStream.close();

    console.log('Downloaded file from S3 to:', localPath);

    // Process the video using video_processor.py
    const pythonCommand = getPythonCommand();
    const pythonProcess = spawn(pythonCommand, [
      path.join(__dirname, 'video_processor.py'),
      localPath,
      localPath  // Use same path for output base name, python script handles suffixes
    ]);

    // Store process reference
    processMap.set(processId, pythonProcess);

    let pythonOutput = '';
    let pythonError = '';

    pythonProcess.stdout.on('data', (data) => {
      const output = data.toString();
      console.log('Python output:', output);
      pythonOutput += output;
    });

    pythonProcess.stderr.on('data', (data) => {
      const error = data.toString();
      console.log('Python error:', error);
      pythonError += error;
    });

    await new Promise((resolve, reject) => {
      pythonProcess.on('close', async (code) => {
        // Clean up process reference
        processMap.delete(processId);

        try {
          if (code !== 0) {
            throw new Error(`Python process failed with code ${code}: ${pythonError}`);
          }

          // Parse Python output
          const matches = pythonOutput.match(/\{(?:[^{}]|(\{[^{}]*\}))*\}/g);
          if (!matches) {
            throw new Error('No valid JSON found in Python output');
          }

          result = JSON.parse(matches[matches.length - 1]);

          if (result.status === 'success') {
            // Read the transcript file safely
            let transcriptContent = '';
            if (result.transcript_file) {
              try {
                // Check if file exists before reading
                await fsPromises.access(result.transcript_file, fs.constants.F_OK);
                transcriptContent = await fsPromises.readFile(result.transcript_file, 'utf8');
              } catch (err) {
                console.warn(`Could not read transcript file (${result.transcript_file}):`, err.message);
                transcriptContent = result.transcript || ''; // Fallback
              }
            } else {
               transcriptContent = result.transcript || ''; // Fallback
            }

            // Generate signed URL for the original video
            const videoCommand = new GetObjectCommand({
              Bucket: process.env.AWS_BUCKET_NAME,
              Key: fileKey
            });

            const originalUrl = await getSignedUrl(s3Client, videoCommand, { expiresIn: 3600 });

            res.json({
              success: true,
              originalUrl: originalUrl,
              data: {
                transcript: transcriptContent,
                segments: result.segments || [],
                summary: result.summary,
                keyPoints: result.keyPoints,
                flashcards: result.flashcards
              }
            });
            resolve();
          } else {
            throw new Error(result.error || 'Processing failed');
          }
        } catch (error) {
          reject(error);
        }
      });
    });
  } catch (error) {
    console.error('Error processing video:', error);
    // Clean up process on error
    cleanupProcess(processId);
    res.status(500).json({
      error: 'Failed to process video',
      details: error.message
    });
  } finally {
    // --- Release Lock ---
    if (fileKey && processingFiles.has(fileKey)) {
      processingFiles.delete(fileKey);
      console.log(`Released lock for key: ${fileKey}`);
    }

    // Clean up temporary files safely
    try {
      // Clean up original video file
      if (localPath) {
        try {
            await fsPromises.access(localPath, fs.constants.F_OK); // Check existence
            await fsPromises.unlink(localPath);
            console.log(`Cleaned up temp video file: ${localPath}`);
        } catch (accessErr) {
            if (accessErr.code !== 'ENOENT') { // Ignore 'file not found'
                console.error(`Error checking/removing temp video file ${localPath}:`, accessErr);
            } else {
                 console.log(`Temp video file ${localPath} already removed.`);
            }
        }
      }

      // Clean up transcript file
      if (result?.transcript_file) {
         try {
            await fsPromises.access(result.transcript_file, fs.constants.F_OK); // Check existence
            await fsPromises.unlink(result.transcript_file);
            console.log(`Cleaned up temp transcript file: ${result.transcript_file}`);
         } catch (accessErr) {
            if (accessErr.code !== 'ENOENT') { // Ignore 'file not found'
                console.error(`Error checking/removing temp transcript file ${result.transcript_file}:`, accessErr);
            } else {
                 console.log(`Temp transcript file ${result.transcript_file} already removed.`);
            }
         }
      }
    } catch (err) {
      console.error('Error in cleanup:', err);
    }
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

// Serve uploaded videos
app.use('/video', express.static(path.join(__dirname, '..', 'uploads')));

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
