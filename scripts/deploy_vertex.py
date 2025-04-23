from google.cloud import aiplatform
import os
from dotenv import load_dotenv

load_dotenv()

def deploy_model(model_path: str = None):
    """
    Deploy the YOLOv8 model to Vertex AI
    
    Args:
        model_path (str): Path to the model file. If None, uses MODEL_PATH from env
    """
    try:
        # Initialize Vertex AI
        project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
        if not project_id:
            raise ValueError("GOOGLE_CLOUD_PROJECT environment variable not set")
            
        aiplatform.init(project=project_id)

        # Create endpoint
        endpoint = aiplatform.Endpoint.create(
            display_name="furniture-detection-endpoint"
        )

        # Load model from path
        model_path = model_path or os.getenv('MODEL_PATH')
        if not model_path:
            raise ValueError("Model path not provided and MODEL_PATH not set in environment")
            
        model = aiplatform.Model.upload(
            display_name="furniture-detection-model",
            artifact_uri=model_path,
            model_id="furniture-detection"
        )

        # Deploy model
        deployed_model = endpoint.deploy(
            model=model,
            machine_type="n1-standard-4",
            min_replica_count=1,
            max_replica_count=3,
            accelerator_type="NVIDIA_TESLA_T4",
            accelerator_count=1
        )

        print(f"Model deployed successfully to endpoint: {endpoint.resource_name}")
        return endpoint
        
    except Exception as e:
        print(f"Error deploying model: {str(e)}")
        raise

if __name__ == "__main__":
    deploy_model() 