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
  origin: true,  // Allow all origins during development
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Accept', 'Authorization'],
  credentials: false  // Change to false since we don't need cookies
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
    const uniqueName = `${Date.now()}-${file.originalname}`;
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

// Serve processed videos
app.use('/videos', express.static(path.join(__dirname, 'uploads', 'processed')));

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
    const audioPath = path.join(__dirname, 'uploads', 'audio', `${filename}.wav`);
    const outputPath = path.join(__dirname, 'uploads', 'processed', filename);

    // Log processing steps
    console.log('Starting audio extraction...');
    await extractAudio(inputPath, audioPath);
    console.log('Audio extraction complete');

    console.log('Starting transcription...');
    const transcript = await simulateTranscription(audioPath);
    console.log('Transcription complete');

    console.log('Starting video processing...');
    await processVideo(inputPath, outputPath, transcript);
    console.log('Video processing complete');

    const videoUrl = `http://100.70.34.122:${PORT}/videos/${filename}`;
    
    // Log success
    console.log('Processing completed successfully:', {
      filename,
      videoUrl
    });

    res.json({
      success: true,
      url: videoUrl,
      filename,
      transcript
    });
  } catch (error) {
    // Detailed error logging
    console.error('Processing error:', {
      error: error.message,
      stack: error.stack,
      file: req.file?.originalname
    });

    // Send appropriate error response
    res.status(500).json({ 
      error: 'Processing failed',
      message: error.message
    });
  } finally {
    // Could add cleanup here if needed
    console.log('Request processing completed');
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

function processVideo(inputPath, outputPath, transcript) {
  return new Promise((resolve, reject) => {
    // For now, just speed up the entire video
    // TODO: Implement smart speed adjustment based on transcript
    ffmpeg(inputPath)
      .videoFilters('setpts=0.5*PTS')
      .on('end', resolve)
      .on('error', reject)
      .save(outputPath);
  });
}

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
