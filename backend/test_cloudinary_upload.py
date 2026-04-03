#!/usr/bin/env python
"""
Example script to test Cloudinary upload functionality
"""
from upload_service import upload_service
import os
from pathlib import Path

def test_upload():
    """Test upload functionality"""
    
    # Create a simple test image (1x1 red pixel PNG)
    test_image_path = "test_image.png"
    test_video_path = "test_video.mp4"
    
    print("=" * 60)
    print("Testing Cloudinary Upload Service")
    print("=" * 60)
    
    # Create a minimal PNG file for testing
    if not os.path.exists(test_image_path):
        print("\n📸 Creating test image...")
        # Minimal red PNG (1x1 pixel)
        png_data = bytes([
            0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a,
            0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52,
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
            0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
            0xde, 0x00, 0x00, 0x00, 0x0c, 0x49, 0x44, 0x41,
            0x54, 0x08, 0x99, 0x63, 0xf8, 0xcf, 0xc0, 0x00,
            0x00, 0x00, 0x03, 0x00, 0x01, 0x3b, 0xb6, 0xee,
            0x56, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4e,
            0x44, 0xae, 0x42, 0x60, 0x82
        ])
        with open(test_image_path, "wb") as f:
            f.write(png_data)
        print(f"✓ Created test image: {test_image_path}")
    
    # Test image upload
    print("\n1️⃣  Testing Image Upload...")
    print("-" * 60)
    try:
        result = upload_service.upload_image(
            test_image_path,
            public_id="test-image",
            folder="test"
        )
        
        if result['success']:
            print(f"✓ Image uploaded successfully!")
            print(f"  URL: {result['url']}")
            print(f"  Public ID: {result['public_id']}")
            print(f"  Size: {result['size']} bytes")
        else:
            print(f"✗ Upload failed: {result.get('error')}")
            print("\n⚠️  Make sure you have set Cloudinary credentials in .env:")
            print("   - CLOUDINARY_CLOUD_NAME")
            print("   - CLOUDINARY_API_KEY")
            print("   - CLOUDINARY_API_SECRET")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Clean up test image
    if os.path.exists(test_image_path):
        os.remove(test_image_path)
    
    print("\n" + "=" * 60)
    print("✓ Test complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Set up Cloudinary account: https://cloudinary.com/users/register/free")
    print("2. Update .env with your credentials")
    print("3. Run this script again to verify upload works")
    print("4. Start using the upload API endpoints!")

if __name__ == "__main__":
    test_upload()
