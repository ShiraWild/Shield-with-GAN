import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

parser = argparse.ArgumentParser()


parser.add_argument("--base_path", type=str, help="base path for extracting logs and saving plots")

args = parser.parse_args()
base_path = args.base_path

"""
costs_log_path = base_path + "/costs_log.csv"
rewards_log_path = base_path + "/rewards_log.csv"
"""
episodes_len_log_path = base_path + "/episodes_len.csv"
shield_losses_updates_path = base_path + "/shield_losses_updates.csv"

#shield_losses_path = base_path + "/shield_losses_updates.csv"

ppo_loss_path = base_path + "/ppo_update_log.csv"
"""
costs_log = pd.read_csv(costs_log_path)
rewards_log = pd.read_csv(rewards_log_path)
shield_losses_updates = pd.read_csv(shield_losses_updates_path)
"""
shield_losses_updates = pd.read_csv(shield_losses_updates_path)

#episodes_len_log = pd.read_csv(episodes_len_log_path)
#shield_losses_log = pd.read_csv(shield_losses_path)

#ppo_losses_log = pd.read_csv(ppo_loss_path)

def create_simple_bar_plot(data, x_axis, y_axis, base_path, figure_name):
    plt.figure(figsize=(10, 6))
    plt.plot(data[x_axis].tolist(), data[y_axis].tolist(), color='green')
    # Adding titles and labels
    plt.title(figure_name)
    plt.grid()  # Optional: add a grid for better readability
    plt.legend()  # Optional: show the legend
    plt.tight_layout()  # Adjust layout to make room for the title and labels
    base_path_plots = base_path + "/plots"
    os.makedirs(base_path_plots, exist_ok=True)
    plt.savefig(base_path_plots + "/" + figure_name + ".png")

if __name__ == '__main__':
    """
    create_simple_bar_plot(costs_log, "Episode", "Cumulative Cost", base_path, "cumulative costs over episodes")
    create_simple_bar_plot(rewards_log, "Time Step", "Reward", base_path, "rewards over time")
    create_simple_bar_plot(shield_losses_updates, "Update Time Step", "Loss", base_path, "shield loss over update timesteps")
    create_simple_bar_plot(episodes_len_log, "Episode", "Episode Length", base_path, "Episode lengths over training")
    """
    #create_simple_bar_plot(episodes_len_log, "Episode", "Episode Length", base_path, "Episode lengths over training")
    #create_simple_bar_plot(ppo_losses_log, "update_timestep", "loss", base_path, "PPO loss over update timesteps")
    create_simple_bar_plot(shield_losses_updates, "Update Time Step", "Loss", base_path, "shield loss over update timesteps")




