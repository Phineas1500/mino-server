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
const PORT = 3001;

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

// Replace your existing /process/s3-video endpoint with this:
app.post('/process/s3-video', async (req, res) => {
  let localPath;
  let result;
  const processId = Date.now().toString();

  // Set up cleanup on client disconnect
  req.on('close', () => {
    if (processMap.has(processId)) {
      console.log('Client disconnected, cleaning up...');
      cleanupProcess(processId);
    }
  });

  try {
    const { fileKey } = req.body;
    
    if (!fileKey) {
      return res.status(400).json({ 
        error: 'fileKey is required' 
      });
    }

    console.log('Processing video from S3:', fileKey);

    // Create temp directory if it doesn't exist
    const tempDir = path.join(__dirname, '..', 'temp');
    await fsPromises.mkdir(tempDir, { recursive: true });

    // Set up path for original video
    localPath = path.join(tempDir, path.basename(fileKey));
    
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
      localPath  // Use same path, Python will append _shortened
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
            // Read the transcript file
            let transcriptContent = '';
            
            if (result.transcript_file) {
              try {
                transcriptContent = await fsPromises.readFile(result.transcript_file, 'utf8');
              } catch (err) {
                console.warn('Could not read transcript file:', err);
              }
            }

            // Upload shortened video to S3 if it exists
            const shortenedKey = `shortened/${path.basename(fileKey)}`;
            if (result.shortened_video_path && await fsPromises.access(result.shortened_video_path).then(() => true).catch(() => false)) {
              await s3Client.send(new PutObjectCommand({
                Bucket: process.env.AWS_BUCKET_NAME,
                Key: shortenedKey,
                Body: await fsPromises.readFile(result.shortened_video_path)
              }));
              console.log('Uploaded shortened video to S3:', shortenedKey);
            }
            
            // Generate signed URLs for both videos
            const videoCommand = new GetObjectCommand({
              Bucket: process.env.AWS_BUCKET_NAME,
              Key: fileKey
            });

            const shortenedVideoCommand = new GetObjectCommand({
              Bucket: process.env.AWS_BUCKET_NAME,
              Key: shortenedKey
            });
            
            const [originalUrl, shortenedUrl] = await Promise.all([
              getSignedUrl(s3Client, videoCommand, { expiresIn: 3600 }),
              getSignedUrl(s3Client, shortenedVideoCommand, { expiresIn: 3600 })
            ]);

            console.log('Generated signed URLs for videos');

            res.json({
              success: true,
              originalUrl: originalUrl,
              shortenedUrl: shortenedUrl,
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
    // Clean up temporary files
    try {
      // Clean up original video file
      if (localPath) {
        await fsPromises.unlink(localPath).catch(err => 
          console.error('Error removing original video temp file:', err)
        );
      }
      
      // Clean up shortened video using the path from Python output
      if (result?.shortened_video_path) {
        await fsPromises.unlink(result.shortened_video_path).catch(err => 
          console.error('Error removing shortened video temp file:', err)
        );
      }
      
      // Clean up transcript file
      if (result?.transcript_file) {
        await fsPromises.unlink(result.transcript_file).catch(err => 
          console.error('Error removing transcript temp file:', err)
        );
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
