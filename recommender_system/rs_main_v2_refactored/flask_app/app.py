import os
import sys
from pathlib import Path
import logging
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from functools import wraps
from typing import Dict, Any, List

# Add parent directory to system path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from encoder_utils import DataEncoder
from generate_recommendations import RecommendationGenerator
from path_config import setup_paths

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ValidationError(Exception):
    pass

def validate_request_data(data: Dict[str, Any]) -> None:
    """Validate incoming request data."""
    required_fields = {
        'age': int,
        'gender': str,
        'favourite_genres': list,
        'favourite_artists': list,
        'favourite_music': list
    }
    
    for field, field_type in required_fields.items():
        if field not in data:
            raise ValidationError(f"Missing required field: {field}")
        if not isinstance(data[field], field_type):
            raise ValidationError(f"Invalid type for {field}: expected {field_type.__name__}")

def validate_request(f):
    """Request validation decorator."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            if not request.is_json:
                return jsonify({'error': 'Content-Type must be application/json'}), 400
            
            data = request.get_json()
            validate_request_data(data)
            return f(*args, **kwargs)
        except ValidationError as e:
            return jsonify({'error': str(e)}), 400
        except Exception as e:
            logger.error(f"Request validation error: {str(e)}")
            return jsonify({'error': 'Internal server error'}), 500
    return decorated_function

class MusicRecommenderAPI:
    def __init__(self):
        self.paths = setup_paths()
        logger.info(f"Initialized paths: {self.paths}")
        
        # Update model paths
        self.model_paths = [
            self.paths['models'] / 'best_model.pth',
            self.paths['models'] / 'best_model_cv.pth',
            self.paths['models'] / 'latest_checkpoint.pt'
        ]
        
        self.data_paths = [
            self.paths['test_data'],
            self.paths['processed_data'] / 'test_data.csv'
        ]
        
        self.encoder_paths = [
            self.paths['encoder'],
            self.paths['processed_data'] / 'encoder.pt'
        ]
        
        self._initialize_system()
        
    def _find_file(self, paths: List[Path], file_type: str) -> Path:
        """Find the first existing file from a list of possible paths."""
        for path in paths:
            if path.exists():
                logger.info(f"Found {file_type} at: {path}")
                return path
                
        # Log all attempted paths
        logger.error(f"Could not find {file_type}. Attempted paths:")
        for path in paths:
            logger.error(f"  - {path}")
        raise FileNotFoundError(f"{file_type} not found")
    
    def _initialize_system(self):
        """Initialize recommender system components with better error handling."""
        try:
            # Validate paths exist
            for path_type, paths in {
                'model': self.model_paths,
                'data': self.data_paths,
                'encoder': self.encoder_paths
            }.items():
                path = self._find_file(paths, f"{path_type} file")
                setattr(self, f"{path_type}_path", path)
                logger.info(f"Using {path_type} from: {path}")
            
            # Load catalog data
            self.catalog_data = pd.read_csv(self.data_path)
            logger.info(f"Loaded catalog with {len(self.catalog_data)} items")
            
            # Add missing columns for compatibility
            if 'main_genre' not in self.catalog_data.columns:
                logger.warning("Adding main_genre column")
                self.catalog_data['main_genre'] = 'Other'
            
            if 'explicit' not in self.catalog_data.columns:
                logger.warning("Adding explicit column")
                self.catalog_data['explicit'] = False
                
            # Initialize recommendation generator
            self.recommender = RecommendationGenerator(
                str(self.model_path),
                self.catalog_data,
                str(self.encoder_path)
            )
            
            logger.info("Music recommender API initialized successfully")
            
        except Exception as e:
            logger.error(f"System initialization failed: {str(e)}", exc_info=True)
            raise
    
    def _prepare_user_data(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare user data for recommendation."""
        try:
            user_data = {
                'age': int(request_data['age']),
                'gender': request_data['gender'].upper(),
                'main_genre': request_data['favourite_genres'][0] if request_data['favourite_genres'] else 'Other',
                'music': request_data['favourite_music'],
                'artist_name': request_data['favourite_artists']
            }
            
            # Add required numerical features with defaults
            for feature in self.recommender.encoders.numerical_features:
                if feature not in user_data and feature != 'age':
                    user_data[feature] = 0.0
                    
            return user_data
            
        except Exception as e:
            logger.error(f"Error preparing user data: {str(e)}")
            raise
    
    def get_recommendations(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate recommendations from request data."""
        try:
            # Prepare user data
            user_data = self._prepare_user_data(request_data)
            
            # Generate recommendations
            recommendations = self.recommender.generate_recommendations(
                user_data,
                n_recommendations=10
            )
            
            # Format response
            response = {
                'status': 'success',
                'recommendations': recommendations.to_dict(orient='records'),
                'metadata': {
                    'user_profile': {
                        'age': user_data['age'],
                        'gender': user_data['gender'],
                        'preferred_genre': user_data['main_genre']
                    }
                }
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Recommendation error: {str(e)}")
            raise

def create_app():
    """Application factory function."""
    app = Flask(__name__)
    CORS(app)
    
    try:
        # Initialize API
        app.config['API'] = MusicRecommenderAPI()
        
        # Register routes
        @app.route('/api/health')
        def health_check():
            """Health check endpoint."""
            try:
                api = app.config['API']
                return jsonify({
                    'status': 'healthy',
                    'model_loaded': hasattr(api, 'recommender'),
                    'catalog_size': len(api.catalog_data) if hasattr(api, 'catalog_data') else 0,
                    'encoders_loaded': hasattr(api.recommender, 'encoders') if hasattr(api, 'recommender') else False,
                    'api_version': '1.0.0',
                    'supported_genres': list(api.recommender.encoders.known_genres) if hasattr(api, 'recommender') and hasattr(api.recommender, 'encoders') else []
                })
            except Exception as e:
                logger.error(f"Health check failed: {str(e)}")
                return jsonify({'status': 'error', 'message': str(e)}), 500

        @app.route('/api/recommendations', methods=['POST'])
        @validate_request
        def get_recommendations():
            """Recommendations endpoint."""
            try:
                api = app.config['API']
                request_data = request.get_json()
                response = api.get_recommendations(request_data)
                return jsonify(response)
            except Exception as e:
                logger.error(f"Recommendation error: {str(e)}")
                return jsonify({
                    'status': 'error',
                    'message': str(e)
                }), 500
                
        return app
        
    except Exception as e:
        logger.error(f"Failed to create application: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    try:
        # Create and configure app
        app = create_app()
        
        # Print startup information
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Python path: {sys.path}")
        logger.info("Starting Flask application...")
        
        # Run the app
        app.run(
            host='127.0.0.1',
            port=5000,
            debug=True,
            use_reloader=False  # Disable reloader to prevent duplicate model loading
        )
        
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}", exc_info=True)
        raise