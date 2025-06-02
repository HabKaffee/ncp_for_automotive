# Prerequisites
Download the Carla Simulator 0.9.15 from github https://github.com/carla-simulator/carla/releases/tag/0.9.15/  
Install requirements
```bash
python3.8 -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
```

# Run the pipeline
## Training
```bash
python -m src.training_script --epochs #epochs --batch-size #batches --output_size #control_signals --units #units --sequence_length #seq_len --lr #leaning_rate --device "cuda|cpu"
```
## Testing
In order to test model run simulator with command
```bash
/path/to/CARLA/root/bin/CarlaUE4.sh -quality-level=Epic
```
or for off-screen rendering
```bash
/path/to/CARLA/root/bin/CarlaUE4.sh -quality-level=Epic -RenderOffScreen
```

Connect to simulator from python notebook (simulator_test.ipynb). Run desired blocks from `test` section.
