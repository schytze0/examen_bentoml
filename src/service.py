import numpy as np
import bentoml
from bentoml.io import JSON
from pydantic import BaseModel
from starlette.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import jwt
from datetime import datetime, timedelta

# Secret key and algorithm for JWT authentication
JWT_SECRET_KEY = "your_jwt_secret_key_here"
JWT_ALGORITHM = "HS256"

# User credentials for authentication
USERS = {
    "user123": "password123",
    "user456": "password456"
}

# Function to create a JWT token
def create_jwt_token(user_id: str):
    expiration = datetime.utcnow() + timedelta(hours=1)
    payload = {"sub": user_id, "exp": expiration}
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return token

# Middleware for JWT Authentication
class JWTAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if request.url.path == "/predict":
            token = request.headers.get("Authorization")
            if not token:
                return JSONResponse(status_code=401, content={"detail": "Missing authentication token"})

            try:
                token = token.split()[1]  # Remove 'Bearer ' prefix
                payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            except jwt.ExpiredSignatureError:
                return JSONResponse(status_code=401, content={"detail": "Token has expired"})
            except jwt.InvalidTokenError:
                return JSONResponse(status_code=401, content={"detail": "Invalid token"})

            request.state.user = payload.get("sub")

        response = await call_next(request)
        return response

# Pydantic model to validate input data
class AdmissionInput(BaseModel):
    gre_score: int
    toefl_score: int
    university_rating: int
    sop: float
    lor: float
    cgpa: float
    research: int

# Get the model from the Model Store
lr_model_runner = bentoml.sklearn.get("lr_model:latest").to_runner()

# Create a service API
admission_service = bentoml.Service("admission_service", runners=[lr_model_runner])

# Add the JWTAuthMiddleware to the service
admission_service.add_asgi_middleware(JWTAuthMiddleware)

# Create an API endpoint for the service
@admission_service.api(input=JSON(), output=JSON(), route="/login")
async def login(credentials: dict) -> dict:
    username = credentials.get("username")
    password = credentials.get("password")

    if username in USERS and USERS[username] == password:
        token = create_jwt_token(username)
        return {"token": token}
    else:
        return JSONResponse(status_code=401, content={"detail": "Invalid credentials"})

# Create an API endpoint for the service
@admission_service.api(input=JSON(pydantic_model=AdmissionInput), output=JSON(), route="/predict")
async def predict(input_data: AdmissionInput, ctx: bentoml.Context) -> dict:
    request = ctx.request
    user = request.state.user if hasattr(request.state, "user") else None

    # Convert input data to a NumPy array
    input_array = np.array([
        input_data.gre_score, input_data.toefl_score, input_data.university_rating,
        input_data.sop, input_data.lor, input_data.cgpa, input_data.research
    ]).reshape(1, -1)

    # Make a prediction
    result = await lr_model_runner.predict.async_run(input_array)

    return {
        "prediction": float(result[0]),
        "user": user
    }

