// src/services/s3.js
const { S3Client, PutObjectCommand } = require('@aws-sdk/client-s3');
const { getSignedUrl } = require('@aws-sdk/s3-request-presigner');

// Validate required environment variables
function validateConfig() {
  const required = ['AWS_REGION', 'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_BUCKET_NAME'];
  const missing = required.filter(key => !process.env[key]);
  
  if (missing.length > 0) {
    throw new Error(`Missing required environment variables: ${missing.join(', ')}`);
  }
  
  return {
    region: process.env.AWS_REGION,
    accessKeyId: process.env.AWS_ACCESS_KEY_ID,
    secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY,
    bucketName: process.env.AWS_BUCKET_NAME
  };
}

// Initialize S3 client
function createS3Client() {
  const config = validateConfig();
  
  return new S3Client({
    region: config.region,
    credentials: {
      accessKeyId: config.accessKeyId,
      secretAccessKey: config.secretAccessKey
    }
  });
}

const s3Client = createS3Client();

async function generatePresignedUrl(fileName, fileType) {
  // Validate bucket name is available
  const bucketName = process.env.AWS_BUCKET_NAME;
  if (!bucketName) {
    throw new Error('AWS_BUCKET_NAME is not configured');
  }

  console.log('Generating presigned URL with config:', {
    bucket: bucketName,
    fileName,
    fileType
  });

  const key = `uploads/${Date.now()}-${fileName}`;
  
  const command = new PutObjectCommand({
    Bucket: bucketName,
    Key: key,
    ContentType: fileType
  });

  try {
    const signedUrl = await getSignedUrl(s3Client, command, {
      expiresIn: 3600
    });

    console.log('Successfully generated presigned URL for:', key);

    return {
      url: signedUrl,
      fields: {
        key
      }
    };
  } catch (error) {
    console.error('Error details:', {
      error: error.message,
      stack: error.stack,
      fileName,
      fileType,
      bucketName
    });
    throw error;
  }
}

module.exports = {
  generatePresignedUrl,
  s3Client
};