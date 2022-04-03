# Bimanual Offline RL

1. Train
    ```commandline
    pyana3is train.py num_envs=100
    ```
2. Load checkpoint
    ```commandline
    pyana3is train.py test=True headless=False checkpoint='runs/DualFrankaCQL/nn/model.pth'
    ```
3. Offline dataset generation
   ```commandline
   pyana3is test.py
   ```