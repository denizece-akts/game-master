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
    
    if games_csv.exists() and reviews_csv.exists():
        print("✅ Dataset found.")
        return

    print("⚠️ Dataset missing. Downloading via kagglehub...")
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        import kagglehub
        
        path = kagglehub.dataset_download("mohamedtarek01234/steam-games-reviews-and-rankings")
        downloaded_path = Path(path)
        
        print(f"Dataset downloaded to cache: {downloaded_path}")
        print(f"Moving files to {output_path}...")
        
        for item in downloaded_path.iterdir():
            if item.is_file():
                destination = output_path / item.name
                print(f"Copying {item.name}...")
                shutil.copy2(item, destination)
            elif item.is_dir():
                 destination = output_path / item.name
                 if destination.exists():
                     shutil.rmtree(destination)
                 shutil.copytree(item, destination)

        print("✅ Dataset setup complete.")
        
    except Exception as e:
        print(f"❌ Failed to download dataset via kagglehub: {e}")
        print("If this is an authentication error, please ensure you are logged in via `kagglehub.login()` or have your credentials set up.")

if __name__ == "__main__":
    ensure_dataset()
