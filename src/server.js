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

    console.log('Starting video processing and transcription...');
    const result = await processVideo(inputPath, outputPath);
    console.log('Video processing and transcription complete');

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
      ...result.transcription_data  // This will include summary, keyPoints, flashcards, and transcript
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
    // First do the speed adjustment with ffmpeg
    const tempPath = outputPath + '_temp.mp4';
    ffmpeg(inputPath)
      .videoFilters('setpts=0.25*PTS')  // 4x speed
      .audioFilters('atempo=2.0,atempo=2.0')  // 4x audio speed
      .on('end', () => {
        // Then process with Python for transcription and enhancement
        const pythonProcess = spawn('python3', [
          path.join(__dirname, 'video_processor.py'),
          tempPath,
          outputPath,
          JSON.stringify({
            // Add any additional parameters here
          })
        ]);

        let pythonOutput = '';
        let pythonError = '';

        pythonProcess.stdout.on('data', (data) => {
          pythonOutput += data.toString();
        });

        pythonProcess.stderr.on('data', (data) => {
          pythonError += data.toString();
        });

        pythonProcess.on('close', (code) => {
          // Clean up temporary file
          fs.unlink(tempPath, (err) => {
            if (err) console.error('Error cleaning up temp file:', err);
          });

          if (code === 0) {
            try {
              const result = JSON.parse(pythonOutput);
              if (result.status === 'success') {
                // Save transcription to a file
                const transcriptionPath = outputPath.replace(/\.[^/.]+$/, '_transcript.txt');
                const transcriptionContent = `Full Transcription:\n=================\n\n${result.transcription.full_text}\n\nSegments with Timestamps:\n=======================\n\n${
                  result.transcription.segments.map(seg => 
                    `[${seg.start.toFixed(2)}s -> ${seg.end.toFixed(2)}s] ${seg.text}`
                  ).join('\n')
                }`;
                
                fs.writeFile(transcriptionPath, transcriptionContent, 'utf8', (err) => {
                  if (err) console.error('Error saving transcription:', err);
                });
                
                resolve({
                  ...result,
                  transcriptionPath
                });
              } else {
                reject(new Error(result.error || 'Python processing failed'));
              }
            } catch (e) {
              reject(new Error('Failed to parse Python output'));
            }
          } else {
            reject(new Error(`Python process failed: ${pythonError}`));
          }
        });
      })
      .on('error', reject)
      .save(tempPath);
  });
}

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
