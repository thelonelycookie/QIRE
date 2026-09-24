# Sync results from server

On the server:

```bash
source .venv/bin/activate
python qire_ai_console.py --export
python qire_ai_console.py --pack-results
```

On your Windows PC:

```powershell
scp -i C:\path\to\private_key.key ubuntu@SERVER_PUBLIC_IP:~/qire_ai_console_v1_0/results/exports/qire_results_*.zip .
```

Alternative with rsync on Linux/macOS:

```bash
rsync -avz -e "ssh -i private_key.key" ubuntu@SERVER_PUBLIC_IP:~/qire_ai_console_v1_0/results/ ./qire_results/
```
