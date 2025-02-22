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
  origin: 'http://localhost:3000'
}));

// Configure storage
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const uploadDir = path.join(__dirname, 'uploads', 'raw');
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
  const fullPath = path.join(__dirname, 'uploads', dir);
  if (!fs.existsSync(fullPath)) {
    fs.mkdirSync(fullPath, { recursive: true });
  }
});

// Serve processed videos
app.use('/videos', express.static(path.join(__dirname, 'uploads', 'processed')));

// Upload and process endpoint
app.post('/upload', upload.single('video'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No file uploaded' });
  }

  const inputPath = req.file.path;
  const filename = path.basename(inputPath);
  const audioPath = path.join(__dirname, 'uploads', 'audio', `${filename}.wav`);
  const outputPath = path.join(__dirname, 'uploads', 'processed', filename);

  try {
    // Extract audio for transcription
    await extractAudio(inputPath, audioPath);

    // TODO: Send audio to whisper for transcription
    // For now, let's simulate transcription
    const transcript = await simulateTranscription(audioPath);

    // Process video based on transcript
    await processVideo(inputPath, outputPath, transcript);

    const videoUrl = `http://100.70.34.122:${PORT}/videos/${filename}`;
    res.json({
      success: true,
      url: videoUrl,
      filename,
      transcript
    });
  } catch (error) {
    console.error('Processing error:', error);
    res.status(500).json({ error: 'Processing failed' });
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
