"""
Cloudinary upload service for storing images and videos
"""
import os
import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv
import logging

load_dotenv()

logger = logging.getLogger(__name__)

# Configure Cloudinary
cloudinary.config(
    cloud_name=os.getenv('CLOUDINARY_CLOUD_NAME'),
    api_key=os.getenv('CLOUDINARY_API_KEY'),
    api_secret=os.getenv('CLOUDINARY_API_SECRET')
)

CLOUDINARY_FOLDER = os.getenv('CLOUDINARY_FOLDER', 'pbl5-violence-detector')


class CloudinaryUploadService:
    """Service for uploading files to Cloudinary"""
    
    @staticmethod
    def upload_image(file_path: str, public_id: str = None, folder: str = None) -> dict:
        """
        Upload image to Cloudinary
        
        Args:
            file_path: Path to local image file
            public_id: Optional custom public ID for the image
            folder: Optional subfolder in Cloudinary
            
        Returns:
            dict with upload result including secure_url and public_id
        """
        try:
            upload_folder = f"{CLOUDINARY_FOLDER}/{folder}" if folder else CLOUDINARY_FOLDER
            
            result = cloudinary.uploader.upload(
                file_path,
                folder=upload_folder,
                public_id=public_id,
                resource_type='image',
                overwrite=True,
                quality='auto',
                fetch_format='auto'
            )
            
            logger.info(f"✓ Image uploaded successfully: {result['public_id']}")
            return {
                "success": True,
                "public_id": result['public_id'],
                "url": result['secure_url'],
                "size": result['bytes']
            }
        except Exception as e:
            logger.error(f"✗ Error uploading image: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @staticmethod
    def upload_video(file_path: str, public_id: str = None, folder: str = None) -> dict:
        """
        Upload video to Cloudinary
        
        Args:
            file_path: Path to local video file
            public_id: Optional custom public ID for the video
            folder: Optional subfolder in Cloudinary
            
        Returns:
            dict with upload result including secure_url and public_id
        """
        try:
            upload_folder = f"{CLOUDINARY_FOLDER}/{folder}" if folder else f"{CLOUDINARY_FOLDER}/videos"
            
            result = cloudinary.uploader.upload(
                file_path,
                folder=upload_folder,
                public_id=public_id,
                resource_type='video',
                overwrite=True,
                quality='auto'
            )
            
            logger.info(f"✓ Video uploaded successfully: {result['public_id']}")
            return {
                "success": True,
                "public_id": result['public_id'],
                "url": result['secure_url'],
                "size": result['bytes'],
                "duration": result.get('duration')
            }
        except Exception as e:
            logger.error(f"✗ Error uploading video: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @staticmethod
    def upload_detection_clip(file_path: str, location: str, timestamp: str) -> dict:
        """
        Upload violence detection clip with organized naming
        
        Args:
            file_path: Path to local video file
            location: Camera location/name
            timestamp: Detection timestamp
            
        Returns:
            dict with upload result
        """
        try:
            # Create organized public ID: violence/location/timestamp
            public_id = f"violence/{location.replace(' ', '_')}/{timestamp}"
            upload_folder = f"{CLOUDINARY_FOLDER}/violence_clips"
            
            result = cloudinary.uploader.upload(
                file_path,
                folder=upload_folder,
                public_id=public_id,
                resource_type='video',
                overwrite=True,
                quality='auto'
            )
            
            logger.info(f"✓ Violence clip uploaded: {result['public_id']}")
            return {
                "success": True,
                "public_id": result['public_id'],
                "url": result['secure_url'],
                "size": result['bytes'],
                "duration": result.get('duration')
            }
        except Exception as e:
            logger.error(f"✗ Error uploading detection clip: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @staticmethod
    def delete_file(public_id: str, resource_type: str = 'image') -> dict:
        """
        Delete file from Cloudinary
        
        Args:
            public_id: Public ID of file to delete
            resource_type: 'image' or 'video'
            
        Returns:
            dict with deletion result
        """
        try:
            result = cloudinary.uploader.destroy(public_id, resource_type=resource_type)
            logger.info(f"✓ File deleted: {public_id}")
            return {
                "success": True,
                "message": f"File {public_id} deleted successfully"
            }
        except Exception as e:
            logger.error(f"✗ Error deleting file: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @staticmethod
    def get_upload_url() -> str:
        """
        Generate unsigned upload URL for direct browser upload
        
        Returns:
            str: Upload URL
        """
        return cloudinary.utils.cloudinary_url(
            CLOUDINARY_FOLDER,
            sign_url=False
        )[0]


# Create singleton instance
upload_service = CloudinaryUploadService()
