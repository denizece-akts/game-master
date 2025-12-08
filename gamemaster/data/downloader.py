import os
import shutil
from pathlib import Path

def ensure_dataset(output_dir: str = "./dataset"):
    """
    Checks if the dataset exists. If not, downloads it using kagglehub and moves files to output_dir.
    """
    output_path = Path(output_dir)
    games_csv = output_path / "games_description.csv"
    reviews_csv = output_path / "steam_game_reviews.csv"
    
    # Simple check: if both main CSVs exist, we assume dataset is ready.
    if games_csv.exists() and reviews_csv.exists():
        print("✅ Dataset found.")
        return

    print("⚠️ Dataset missing. Downloading via kagglehub...")
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        import kagglehub
        
        # Download latest version
        # This downloads to a cache directory managed by kagglehub
        path = kagglehub.dataset_download("mohamedtarek01234/steam-games-reviews-and-rankings")
        downloaded_path = Path(path)
        
        print(f"Dataset downloaded to cache: {downloaded_path}")
        print(f"Moving files to {output_path}...")
        
        # Move/Copy files from cache to our dataset folder
        for item in downloaded_path.iterdir():
            if item.is_file():
                destination = output_path / item.name
                # We use copy to keep the cache intact or move? 
                # kagglehub manages cache, so copy is safer, but move saves space if we don't care about cache.
                # Let's use shutil.move or copy. Copy is safer to avoid breaking kagglehub's internal state if it tracks files.
                # However, for a one-off setup, copy is fine.
                print(f"Copying {item.name}...")
                shutil.copy2(item, destination)
            elif item.is_dir():
                 # Recursive copy if needed, but this dataset is known to be flat CSVs usually.
                 destination = output_path / item.name
                 if destination.exists():
                     shutil.rmtree(destination)
                 shutil.copytree(item, destination)

        print("✅ Dataset setup complete.")
        
    except Exception as e:
        print(f"❌ Failed to download dataset via kagglehub: {e}")
        # Optional: Print instructions if it's an auth error, though kagglehub for public datasets often works without auth.
        print("If this is an authentication error, please ensure you are logged in via `kagglehub.login()` or have your credentials set up.")

if __name__ == "__main__":
    ensure_dataset()
