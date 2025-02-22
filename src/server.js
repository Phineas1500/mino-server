// server.js
const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const cors = require('cors');
const { spawn } = require('child_process');

const app = express();
const PORT = 3001;

app.use(cors({
  origin: true,
  methods: ['GET', 'POST'],
  allowedHeaders: ['Content-Type', 'Accept'],
  credentials: false
}));

app.use(express.json());

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
    
    const pythonProcess = spawn('python3', [
      path.join(__dirname, 'video_processor.py'),
      inputPath,
      outputPath
    ]);

    let pythonOutput = '';
    let pythonError = '';

    pythonProcess.stdout.on('data', (data) => {
      const output = data.toString();
      console.log('Python output:', output);
      // Append all output
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
              segments: result.segments
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

// Test endpoint
app.post('/api/transcript/test', (req, res) => {
  const pythonProcess = spawn('python3', [
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
