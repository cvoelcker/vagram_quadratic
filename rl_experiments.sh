# Pendulum Comparison
# python examples/train.py --model_loss_fn vagram_full --max_steps 50000 --seed 0 --model_hidden_size 8
# python examples/train.py --model_loss_fn vagram_full --max_steps 50000 --seed 1 --model_hidden_size 8
# python examples/train.py --model_loss_fn vagram_full --max_steps 50000 --seed 2 --model_hidden_size 8
# python examples/train.py --model_loss_fn vagram_full --max_steps 50000 --seed 3 --model_hidden_size 8
# python examples/train.py --model_loss_fn vagram_full --max_steps 50000 --seed 4 --model_hidden_size 8
# python examples/train.py --model_loss_fn vagram_full --max_steps 50000 --seed 5 --model_hidden_size 8

# python examples/train.py --model_loss_fn vagram --max_steps 50000 --seed 0 --model_hidden_size 8
# python examples/train.py --model_loss_fn vagram --max_steps 50000 --seed 1 --model_hidden_size 8
# python examples/train.py --model_loss_fn vagram --max_steps 50000 --seed 2 --model_hidden_size 8
# python examples/train.py --model_loss_fn vagram --max_steps 50000 --seed 3 --model_hidden_size 8
# python examples/train.py --model_loss_fn vagram --max_steps 50000 --seed 4 --model_hidden_size 8
# python examples/train.py --model_loss_fn vagram --max_steps 50000 --seed 5 --model_hidden_size 8

# python examples/train.py --model_loss_fn mse --max_steps 50000 --seed 0 --model_hidden_size 8
# python examples/train.py --model_loss_fn mse --max_steps 50000 --seed 1 --model_hidden_size 8
# python examples/train.py --model_loss_fn mse --max_steps 50000 --seed 2 --model_hidden_size 8
# python examples/train.py --model_loss_fn mse --max_steps 50000 --seed 3 --model_hidden_size 8
# python examples/train.py --model_loss_fn mse --max_steps 50000 --seed 4 --model_hidden_size 8
# python examples/train.py --model_loss_fn mse --max_steps 50000 --seed 5 --model_hidden_size 8


# Ant comparison
sbatch run_ant.sh vagram_full 0
sbatch run_ant.sh vagram_full 1
sbatch run_ant.sh vagram_full 2
sbatch run_ant.sh vagram_full 3
sbatch run_ant.sh vagram_full 4
sbatch run_ant.sh vagram_full 5

sbatch run_ant.sh vagram 0
sbatch run_ant.sh vagram 1
sbatch run_ant.sh vagram 2
# sbatch run_ant.sh vagram 3
# sbatch run_ant.sh vagram 4
# sbatch run_ant.sh vagram 5
# 
# sbatch run_ant.sh mse 0
# sbatch run_ant.sh mse 1
# sbatch run_ant.sh mse 2
# sbatch run_ant.sh mse 3
# sbatch run_ant.sh mse 4
# sbatch run_ant.sh mse 5
