#!/usr/bin/env python
"""
Khởi tạo database và tạo tất cả các bảng từ models
"""
import sys
from database import engine, Base, init_db
# Import models for metadata registration
import models

if __name__ == "__main__":
    try:
        print("Starting database initialization...")
        init_db()
        print("\nDatabase initialized successfully!")
        print("All tables created from models")
        
        # Verify tables
        from sqlalchemy import inspect
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        print(f"\nTables created: {tables}")
        
    except Exception as e:
        print(f"\nError during database initialization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
