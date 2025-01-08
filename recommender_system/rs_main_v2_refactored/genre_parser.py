import re
from typing import Dict, List, Tuple
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class GenreParser:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.genre_data = defaultdict(list)
        self.parse_file()  # Call parse_file during initialization
        logger.info(f"Parsed genres for categories: {list(self.genre_data.keys())}")
        
    def parse_file(self) -> Dict[str, List[Tuple[str, int]]]:
        """Parse all_genres.txt and extract genre hierarchies with counts"""
        current_category = None
        logger.info(f"Starting to parse file: {self.filepath}")
        
        with open(self.filepath, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Match category headers like "## ROCK (Total: 1865)"
                category_match = re.match(r'^## ([A-Z_]+) \(Total: (\d+)\)', line)
                if category_match:
                    current_category = category_match.group(1).lower()
                    logger.debug(f"Found category: {current_category}")
                    continue
                    
                # Match genre entries like "- rock: 1089"
                genre_match = re.match(r'^- (.+): (\d+)$', line)
                if genre_match and current_category:
                    subgenre = genre_match.group(1).lower()
                    count = int(genre_match.group(2))
                    self.genre_data[current_category].append((subgenre, count))
                    logger.debug(f"Added subgenre '{subgenre}' with count {count} to category '{current_category}'")
        
        logger.info(f"Completed parsing file. Total categories parsed: {len(self.genre_data)}")
        return dict(self.genre_data)
        
    def get_main_genres_dict(self) -> Dict[str, List[str]]:
        """Convert parsed data to MAIN_GENRES format"""
        main_genres = {}
        
        for category, genres in self.genre_data.items():
            # Sort genres by count descending
            sorted_genres = sorted(genres, key=lambda x: x[1], reverse=True)
            # Extract just the genre names
            main_genres[category] = [genre[0] for genre in sorted_genres]
            logger.debug(f"Category '{category}' has genres: {main_genres[category]}")
        
        # Ensure main genres are included as subgenres for exact matching
        for main_genre in main_genres.keys():
            if main_genre not in main_genres[main_genre]:
                main_genres[main_genre].insert(0, main_genre)
                logger.debug(f"Inserted main genre '{main_genre}' into its own subgenres.")
        
        # Capitalize main genres to match encoder_utils expectations
        for main_genre, subgenres in main_genres.items():
            main_genres[main_genre] = [subgenre.title() for subgenre in subgenres]
            logger.debug(f"Capitalized subgenres for main genre '{main_genre}': {main_genres[main_genre]}")
        
        logger.info(f"Main genres dictionary created with {len(main_genres)} categories.")
        return main_genres
