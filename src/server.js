// server.js
const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const cors = require('cors');
const ffmpeg = require('fluent-ffmpeg');
const { spawn } = require('child_process');

const app = express();
const PORT = 3001;

app.use(cors({
  origin: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Accept', 'Authorization', 'Range'],
  exposedHeaders: ['Content-Range', 'Content-Length', 'Accept-Ranges'],
  credentials: false
}));

// Configure storage
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const uploadDir = path.join(__dirname, '..', 'uploads', 'raw');
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
    }
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    // Remove spaces and special characters from original filename
    const cleanFileName = file.originalname.replace(/[^a-zA-Z0-9.]/g, '-');
    const uniqueName = `${Date.now()}-${cleanFileName}`;
    cb(null, uniqueName);
  }
});

const upload = multer({ storage });

// Create directories for different stages
const dirs = ['raw', 'processed', 'audio'];
dirs.forEach(dir => {
  const fullPath = path.join(__dirname, '..', 'uploads', dir);
  if (!fs.existsSync(fullPath)) {
    fs.mkdirSync(fullPath, { recursive: true });
  }
});

// Add MIME type mapping
const mimeTypes = {
  '.mp4': 'video/mp4',
  '.mov': 'video/quicktime',
  '.webm': 'video/webm'
};

// Keep track of seen videos to avoid duplicate logs
const seenVideos = new Set();

// Update static file serving
app.use('/videos', (req, res, next) => {
  const ext = path.extname(req.url).toLowerCase();
  if (mimeTypes[ext]) {
    res.set('Content-Type', mimeTypes[ext]);
    res.set('Accept-Ranges', 'bytes');
    
    // Only log the first time we see a video
    if (!seenVideos.has(req.url)) {
      console.log(`New video request: ${req.url} (${mimeTypes[ext]})`);
      seenVideos.add(req.url);
      
      // Optional: Clear old entries if the set gets too large
      if (seenVideos.size > 1000) {
        seenVideos.clear();
      }
    }
  }
  next();
}, express.static(path.join(__dirname, '..', 'uploads', 'processed')));

// Upload and process endpoint
app.post('/upload', upload.single('video'), async (req, res) => {
  try {
    console.log('Received upload request');

    if (!req.file) {
      console.log('No file received');
      return res.status(400).json({ error: 'No file uploaded' });
    }

    console.log('File received:', req.file.originalname);
    
    const inputPath = req.file.path;
    const filename = path.basename(inputPath);
    const audioPath = path.join(__dirname, '..', 'uploads', 'audio', `${filename}.wav`);
    const outputPath = path.join(__dirname, '..', 'uploads', 'processed', filename);

    // Log processing steps
    console.log('Starting audio extraction...');
    await extractAudio(inputPath, audioPath);
    console.log('Audio extraction complete');

    console.log('Starting video processing and transcription...');
    const result = await processVideo(inputPath, outputPath);
    console.log('Video processing and transcription complete');

    // Use MP4 extension for the URL
    const mp4Filename = filename.replace(/\.[^.]+$/, '.mp4');
    const videoUrl = `http://100.70.34.122:${PORT}/videos/${mp4Filename}`;
    
    console.log('Processing completed successfully:', {
      filename: mp4Filename,
      videoUrl
    });

    res.json({
      success: true,
      url: videoUrl,
      filename: mp4Filename,
      ...result.transcription_data
    });
  } catch (error) {
    console.error('Processing error:', {
      error: error.message,
      stack: error.stack,
      file: req.file?.originalname
    });

    res.status(500).json({ 
      error: 'Processing failed',
      message: error.message
    });
  }
});

function extractAudio(inputPath, outputPath) {
  return new Promise((resolve, reject) => {
    ffmpeg(inputPath)
      .toFormat('wav')
      .audioBitrate(16)
      .audioChannels(1)
      .audioFrequency(16000)
      .on('end', resolve)
      .on('error', reject)
      .save(outputPath);
  });
}

// Temporary function until we integrate with Whisper
function simulateTranscription(audioPath) {
  return new Promise(resolve => {
    setTimeout(() => {
      resolve({
        segments: [
          { start: 0, end: 30, text: "Introduction" },
          { start: 30, end: 60, text: "Main content" },
          // Add more segments as needed
        ]
      });
    }, 1000);
  });
}

function processVideo(inputPath, outputPath) {
  return new Promise((resolve, reject) => {
    const ext = path.extname(inputPath).toLowerCase();
    // Convert outputPath to mp4 and ensure correct path
    const mp4OutputPath = outputPath.replace(/\.[^.]+$/, '.mp4');
    
    console.log('Processing paths:', {
      inputPath,
      outputPath,
      mp4OutputPath
    });

    // Function to run Python processing (only for transcription)
    const runPythonProcessing = (videoPath) => {
      const pythonScript = path.join(__dirname, 'video_processor.py');
      let pythonCommand = 'python3';
      
      const venvPythonPaths = [
        path.join(__dirname, '..', 'venv', 'bin', 'python'),
        path.join(__dirname, '..', 'bin', 'python'),
        path.join(__dirname, '..', 'env', 'bin', 'python'),
        path.join(__dirname, '..', 'venv', 'Scripts', 'python.exe'),
        path.join(__dirname, '..', 'env', 'Scripts', 'python.exe')
      ];

      for (const venvPath of venvPythonPaths) {
        if (fs.existsSync(venvPath)) {
          pythonCommand = venvPath;
          break;
        }
      }

      console.log('Starting transcription process...');

      const pythonProcess = spawn(pythonCommand, [
        pythonScript,
        videoPath,
        'dummy.mp4',  // Dummy output path since we're not using it
        JSON.stringify({ transcriptionOnly: true })  // This should be the third argument
      ]);

      let stdoutData = '';
      let stderrData = '';

      pythonProcess.stdout.on('data', (data) => {
        stdoutData += data.toString();
      });

      pythonProcess.stderr.on('data', (data) => {
        stderrData += data.toString();
        console.log('Python progress:', data.toString());
      });

      return new Promise((resolveTranscription, rejectTranscription) => {
        pythonProcess.on('close', (code) => {
          if (code === 0) {
            try {
              const lastLine = stdoutData.trim().split('\n').pop();
              resolveTranscription(JSON.parse(lastLine));
            } catch (e) {
              console.error('Failed to parse Python output:', e);
              resolveTranscription({
                summary: "",
                keyPoints: [],
                flashcards: [],
                transcript: "",
                segments: []
              });
            }
          } else {
            console.error('Python process failed with code:', code);
            resolveTranscription({
              summary: "",
              keyPoints: [],
              flashcards: [],
              transcript: "",
              segments: []
            });
          }
        });
      });
    };

    // If it's a MOV file, convert to MP4 first with better encoding settings
    if (ext === '.mov') {
      console.log('Converting MOV to MP4...');
      ffmpeg(inputPath)
        .toFormat('mp4')
        .outputOptions(
          '-c:v', 'libx264',     // Video codec
          '-preset', 'medium',    // Encoding speed preset
          '-crf', '23',          // Quality (lower = better, 18-28 is good)
          '-profile:v', 'high',   // H.264 profile
          '-level', '4.0',       // H.264 level
          '-movflags', '+faststart',  // Enable streaming
          '-c:a', 'aac',         // Audio codec
          '-b:a', '128k',        // Audio bitrate
          '-pix_fmt', 'yuv420p'  // Pixel format for better compatibility
        )
        .on('progress', (progress) => {
          console.log('FFmpeg Progress:', progress);
        })
        .on('end', async () => {
          console.log('MOV to MP4 conversion complete');
          try {
            // Get transcription data but don't modify the video
            const transcriptionData = await runPythonProcessing(inputPath);
            resolve({
              status: "success",
              frames_processed: 0,
              fps: 30.0,
              dimensions: "1280x720",
              transcription_data: transcriptionData
            });
          } catch (error) {
            console.error('Transcription error:', error);
            // Still resolve with success even if transcription fails
            resolve({
              status: "success",
              frames_processed: 0,
              fps: 30.0,
              dimensions: "1280x720",
              transcription_data: {
                summary: "",
                keyPoints: [],
                flashcards: [],
                transcript: "",
                segments: []
              }
            });
          }
        })
        .on('error', (err) => {
          console.error('FFmpeg error:', err);
          reject(new Error('Failed to convert video format'));
        })
        .save(mp4OutputPath);
    } else {
      // For non-MOV files, just ensure proper MP4 format
      ffmpeg(inputPath)
        .outputOptions(
          '-c', 'copy',  // Just copy streams without re-encoding if possible
          '-movflags', '+faststart'  // Enable streaming
        )
        .save(mp4OutputPath)
        .on('end', async () => {
          try {
            // Get transcription data but don't modify the video
            const transcriptionData = await runPythonProcessing(inputPath);
            resolve({
              status: "success",
              frames_processed: 0,
              fps: 30.0,
              dimensions: "1280x720",
              transcription_data: transcriptionData
            });
          } catch (error) {
            console.error('Transcription error:', error);
            // Still resolve with success even if transcription fails
            resolve({
              status: "success",
              frames_processed: 0,
              fps: 30.0,
              dimensions: "1280x720",
              transcription_data: {
                summary: "",
                keyPoints: [],
                flashcards: [],
                transcript: "",
                segments: []
              }
            });
          }
        })
        .on('error', reject);
    }
  });
}

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
