# TyreTrack V5 — build on GitHub Actions (Windows)

This repository contains the V5 source and detector. It does **not** contain a compiled EXE.

## Upload (important: preserve the hidden `.github` directory)

1. Create a new **private** GitHub repository.
2. Upload all files and folders from this ZIP **to the repository root**, including `.github/workflows/windows-v5.yml`. The path must be EXACTLY `.github/workflows/windows-v5.yml` — no spaces, no extra `github/` directory, no `.txt` extension.
3. GitHub's web uploader sometimes handles hidden folders awkwardly. The reliable method is GitHub Desktop: extract ZIP, add the extracted directory as a local repository, publish it as **private**. Alternatively use the GitHub web editor: **Add file → Create new file** and enter `.github/workflows/windows-v5.yml`, then paste the workflow content.
4. In GitHub, open **Actions → TyreTrack V5 Windows Portable → Run workflow** (select `main`).
5. Wait for a green check. Open the run, scroll to **Artifacts**, and download `TyreTrack-V5-Windows-Portable`.
6. Extract the artifact ZIP. Copy the **whole** `TyreTrack_V5_<timestamp>` directory to the Windows PC. Start `TyreTrack.exe` there.

## What the workflow does

- Uses GitHub's Windows runner, Python 3.12 x64, CPU-only PyTorch.
- Runs the 16-case synthetic report test, verifies detector model and dependencies.
- Builds with the previous working build strategy, relocates duplicate Qt runtime DLLs when safe, and preserves PyTorch's required DLLs.
- Runs a 10-second EXE startup smoke test before uploading the portable folder.

## Important limitations

- A green build does **not** prove real OAK-D camera, video counts, SMTP email, or GUI controls on your Australian PC. Test those before production.
- If the build fails, download `TyreTrack-V5-build-logs` and inspect the Actions job error.
- The included detector is `tyre_detector.pt`. A separate classifier model is not included.
- **Never upload** email passwords, `email_config.ini`, private databases or working production data.
- The existing V4/V5 working application is not touched.
- GitHub hosted Windows runner and the target Windows PC can have different installed DLLs; test the downloaded EXE locally.
