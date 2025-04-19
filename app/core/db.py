from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from app.core.config import settings
from aiohttp import ClientSession, TCPConnector
from beanie import init_beanie

# Connect to MongoDB
class MongoDBConnection:
    def __init__(self) -> None:
        self.client = None
        self.db = None
        self.session = None

    async def connect(self) -> None:
        try:
            # TCP Session
            # proxy = settings.HTTP_PROXY
            # if proxy != "nil":
            #     connector = TCPConnector()
            #     self.session = ClientSession(connector=connector)
            #     self.client = AsyncIOMotorClient(
            #         host=settings.MONGO_URI,
            #         serverSelectionTimeoutMS=5000,
            #         proxy=proxy
            #     )
            # else:
            self.client = AsyncIOMotorClient(settings.MONGO_URI)

            self.db: AsyncIOMotorDatabase = self.client[settings.MONGO_DATABASE]
            await self.db.command("ping")
            
            # Initialize Beanie with document models
            await self.init_beanie()
            
            print("Successfully connected to MongoDB and initialized Beanie.")
        except Exception as e:
            print(f"Failed to connect to MongoDB: {e}")

    async def init_beanie(self) -> None:
        """Initialize Beanie with all document models"""
        try:
            # Import all document models here -> Avoid circular inpoter
            from app.models.chatModel import ChatModel
            # Add other document models as needed
            
            # Initialize Beanie with the database and document models
            await init_beanie(
                database=self.db,
                document_models=[
                    ChatModel,
                    # Add other document models here
                ]
            )
            print("Beanie initialized successfully.")
        except Exception as e:
            print(f"Failed to initialize Beanie: {e}")
            raise

    async def close(self) -> None:
        if self.client:
            self.client.close()
        if self.session:
            await self.session.close()
        print("MongoDB connection closed.")

# Khởi tạo đối tượng MongoDBConnection
mongo_connection = MongoDBConnection()