from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from app.core.config import settings

# Connect to MongoDB
class MongoDBConnection:
    def __init__(self) -> None:
        self.client = None
        self.db = None

    async def connect(self) -> None:
        try:
            self.client = AsyncIOMotorClient(settings.MONGO_URI)
            self.db: AsyncIOMotorDatabase = self.client[settings.MONGO_DATABASE]
            await self.db.command("ping")
            print("Successfully connected to MongoDB.")
        except Exception as e:
            print(f"Failed to connect to MongoDB: {e}")

    async def close(self) -> None:
        if self.client:
            self.client.close()
            print("MongoDB connection closed.")

# Khởi tạo đối tượng MongoDBConnection
mongo_connection = MongoDBConnection()

