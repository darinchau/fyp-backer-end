import os
import uvicorn
from app import create_app

if __name__ == '__main__':
    uvicorn.run(create_app(), port=os.getenv('PORT', 8123))
