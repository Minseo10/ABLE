"""Make a code/TidyBot-model deployment archive; never include Sprout assets."""

import json
import sys
import tarfile
from pathlib import Path


def package(repo: Path, output: Path) -> None:
    tidybot = repo.parent / "prpl-tidybot"
    sys.path.insert(0, str(tidybot / "src"))
    from prpl_tidybot.sprout.live_config import model_files

    # Ship the cohesive original pipeline as well as Sprout adapters. Only
    # Python source/public TidyBot assets are included, never private models.
    files = [tidybot / "hardware_tests/test_arm_retract.py"]
    files.extend((tidybot / "src/prpl_tidybot").rglob("*.py"))
    files.append(
        tidybot / "src/prpl_tidybot/third_party/models/gen3_robotiq_2f_85.urdf"
    )
    scene = json.loads((repo / ".local/sprout/scene.json").read_text())
    model = Path(scene["tidybot_model"]).resolve()
    if model.name != "tidybot.xml" or model.parent.name != "stanford_tidybot":
        raise ValueError("Only the public stanford_tidybot model can be packaged")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as archive:
        for file in sorted(files):
            archive.add(file, arcname=str(file.relative_to(tidybot)), recursive=False)
        for name, file in sorted(model_files(model).items()):
            archive.add(
                file, arcname=str(Path("tidybot-model") / name), recursive=False
            )
    print(f"Saved {output}: code + public TidyBot model only", flush=True)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    package(root, root / ".local/sprout/deploy.tar.gz")
