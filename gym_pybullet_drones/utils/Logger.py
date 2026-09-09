import os
from datetime import datetime
from cycler import cycler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

os.environ['KMP_DUPLICATE_LIB_OK']='True'

class Logger(object):
    """A class for logging and visualization.

    Stores, saves to file, and plots the kinematic information and RPMs
    of a simulation with one or more drones.

    """

    ################################################################################

    def __init__(self,
                 logging_freq_hz: int,
                 output_folder: str="results",
                 num_drones: int=1,
                 duration_sec: int=0,
                 colab: bool=False,
                 ):
        """Logger class __init__ method.

        Note: the order in which information is stored by Logger.log() is not the same
        as the one in, e.g., the obs["id"]["state"], check the implementation below.

        Parameters
        ----------
        logging_freq_hz : int
            Logging frequency in Hz.
        num_drones : int, optional
            Number of drones.
        duration_sec : int, optional
            Used to preallocate the log arrays (improves performance).

        """
        self.COLAB = colab
        self.OUTPUT_FOLDER = output_folder
        if not os.path.exists(self.OUTPUT_FOLDER):
            os.mkdir(self.OUTPUT_FOLDER)
        self.LOGGING_FREQ_HZ = logging_freq_hz
        self.NUM_DRONES = num_drones
        self.PREALLOCATED_ARRAYS = False if duration_sec == 0 else True
        self.counters = np.zeros(num_drones)
        self.timestamps = np.zeros((num_drones, duration_sec*self.LOGGING_FREQ_HZ))
        #### Note: this is the suggest information to log ##############################
        self.actions = np.zeros((num_drones, 4, duration_sec*self.LOGGING_FREQ_HZ))
        self.obs = np.zeros((num_drones, 12, duration_sec*self.LOGGING_FREQ_HZ)) #### 16 states: pos_x,
                                                                                                  # pos_y,
                                                                                                  # pos_z,
                                                                                                  # vel_x,
                                                                                                  # vel_y,
                                                                                                  # vel_z,
                                                                                                  # roll,
                                                                                                  # pitch,
                                                                                                  # yaw,
                                                                                                  # ang_vel_x,
                                                                                                  # ang_vel_y,
                                                                                                  # ang_vel_z,
                                                                                                  # rpm0,
                                                                                                  # rpm1,
                                                                                                  # rpm2,
                                                                                                  # r
        self.rewards = np.zeros((num_drones, duration_sec*self.LOGGING_FREQ_HZ))
        #### Note: this is the suggest information to log ##############################
        self.controls = np.zeros((num_drones, 12, duration_sec*self.LOGGING_FREQ_HZ)) #### 12 control targets: pos_x,
                                                                                                             # pos_y,
                                                                                                             # pos_z,
                                                                                                             # vel_x, 
                                                                                                             # vel_y,
                                                                                                             # vel_z,
                                                                                                             # roll,
                                                                                                             # pitch,
                                                                                                             # yaw,
                                                                                                             # ang_vel_x,
                                                                                                             # ang_vel_y,
                                                                                                             # ang_vel_z

    ################################################################################

    def log(self,
            drone: int,
            timestamp,
            action,
            obs,
            reward,
            control=np.zeros(12)
            ):
        """Logs entries for a single simulation step, of a single drone.

        Parameters
        ----------
        drone : int
            Id of the drone associated to the log entry.
        timestamp : float
            Timestamp of the log in simulation clock.
        state : ndarray
            (20,)-shaped array of floats containing the drone's state.
        control : ndarray, optional
            (12,)-shaped array of floats containing the drone's control target.

        """
        if drone < 0 or drone >= self.NUM_DRONES or timestamp < 0 or len(obs) != 12 or len(control) != 12:
            print("[ERROR] in Logger.log(), invalid data")
        current_counter = int(self.counters[drone])
        #### Add rows to the matrices if a counter exceeds their size
        if current_counter >= self.timestamps.shape[1]:
            self.timestamps = np.concatenate((self.timestamps, np.zeros((self.NUM_DRONES, 1))), axis=1)
            self.actions = np.concatenate((self.actions, np.zeros((self.NUM_DRONES, 4, 1))), axis=2)
            self.obs = np.concatenate((self.obs, np.zeros((self.NUM_DRONES, 12, 1))), axis=2)
            self.rewards = np.concatenate((self.rewards, np.zeros((self.NUM_DRONES, 1))), axis=1)
            self.controls = np.concatenate((self.controls, np.zeros((self.NUM_DRONES, 12, 1))), axis=2)
        #### Advance a counter is the matrices have overgrown it ###
        elif not self.PREALLOCATED_ARRAYS and self.timestamps.shape[1] > current_counter:
            current_counter = self.timestamps.shape[1]-1
        #### Log the information and increase the counter ##########
        self.timestamps[drone, current_counter] = timestamp
        #### Re-order the kinematic obs (of most Aviaries) #########
        self.actions[drone, :, current_counter] = action
        self.obs[drone, :, current_counter] = obs
        self.rewards[drone, current_counter] = reward
        self.controls[drone, :, current_counter] = control
        self.counters[drone] = current_counter + 1

    ################################################################################

    def load(self, filename):
        with np.load(os.path.join(self.OUTPUT_FOLDER, filename + ".npz")) as data:
            self.timestamps = data["timestamps"]
            self.obs = data["obs"]
            self.actions = data["actions"]
            self.rewards = data["rewards"]
            self.NUM_DRONES = self.timestamps.shape[0]

    ################################################################################

    def save(self,
                    comment: str=""
                    ):
        """Save the logs---on your Desktop---as comma separated values.

        Parameters
        ----------
        comment : str, optional
            Added to the foldername.

        """
        df = pd.DataFrame()
        df["timestamps"] = np.arange(0, self.timestamps.shape[1]/self.LOGGING_FREQ_HZ, 1/self.LOGGING_FREQ_HZ)

        for i in range(self.NUM_DRONES):
            df["x"+str(i)] = self.obs[i, 0]
            df["y"+str(i)] = self.obs[i, 1]
            df["z"+str(i)] = self.obs[i, 2]
            df["r"+str(i)] = self.obs[i, 3]
            df["p"+str(i)] = self.obs[i, 4]
            df["ya"+str(i)] = self.obs[i, 5]
            df["vx"+str(i)] = self.obs[i, 6]
            df["vy"+str(i)] = self.obs[i, 7]
            df["vz"+str(i)] = self.obs[i, 8]
            df["wx"+str(i)] = self.obs[i, 9]
            df["wy"+str(i)] = self.obs[i, 10]
            df["wz"+str(i)] = self.obs[i, 11]
            df["rr"+str(i)] = np.hstack([0, (self.obs[i, 3, 1:] - self.obs[i, 3, 0:-1]) * self.LOGGING_FREQ_HZ])
            df["pr"+str(i)] = np.hstack([0, (self.obs[i, 4, 1:] - self.obs[i, 4, 0:-1]) * self.LOGGING_FREQ_HZ])
            df["yar"+str(i)] = np.hstack([0, (self.obs[i, 5, 1:] - self.obs[i, 5, 0:-1]) * self.LOGGING_FREQ_HZ])
            df["ROLL"+str(i)] = self.actions[i, 0]
            df["PITCH"+str(i)] = self.actions[i, 1]
            df["YAW"+str(i)] = self.actions[i, 2]
            df["THRUST"+str(i)] = self.actions[i, 3]
            df["rew"+str(i)] = self.rewards[i]

        filename = "flight-data"
        if comment:
            filename += "-" + comment
        df.to_csv(os.path.join(self.OUTPUT_FOLDER, filename + ".csv"), index=False)

        with open(os.path.join(self.OUTPUT_FOLDER, filename + ".npz"), "wb") as file:
            np.savez(file, timestamps=self.timestamps, obs=self.obs, actions=self.actions, rewards=self.rewards)

    ################################################################################
    
    def plot(self, show=False, pwm=False):
        """Logs entries for a single simulation step, of a single drone.

        Parameters
        ----------
        pwm : bool, optional
            If True, converts logged RPM into PWM values (for Crazyflies).

        """
        #### Loop over colors and line styles ######################
        plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y']) + cycler('linestyle', ['-', '--', ':', '-.'])))
        fig, axs = plt.subplots(10, 2)
        t = np.arange(0, self.timestamps.shape[1]/self.LOGGING_FREQ_HZ, 1/self.LOGGING_FREQ_HZ)

        #### Column ################################################
        col = 0

        #### XYZ ###################################################
        row = 0
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.obs[j, 0, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('x (m)')
        axs[row, col].set_ylim([-0.5, 0.5])

        row = 1
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.obs[j, 1, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('y (m)')
        axs[row, col].set_ylim([-0.5, 0.5])

        row = 2
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.obs[j, 2, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('z (m)')
        axs[row, col].set_ylim([-0.5, 0.5])

        #### RPY ###################################################
        row = 3
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.obs[j, 3, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('r (rad)')
        axs[row, col].set_ylim([-0.4, 0.4])
        row = 4
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.obs[j, 4, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('p (rad)')
        axs[row, col].set_ylim([-0.4, 0.4])
        row = 5
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.obs[j, 5, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('y (rad)')
        axs[row, col].set_ylim([-3.14, 3.14])

        #### Ang Vel ###############################################
        row = 6
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.obs[j, 9, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('wx')
        axs[row, col].set_ylim([-1.0, 1.0])
        row = 7
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.obs[j, 10, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('wy')
        axs[row, col].set_ylim([-1.0, 1.0])
        row = 8
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.obs[j, 11, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('wz')
        axs[row, col].set_ylim([-3.14, 3.14])

        #### Time ##################################################
        row = 9
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.rewards[j], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('reward')
        axs[row, col].set_ylim([0.0, 1.5])

        #### Column ################################################
        col = 1

        #### Velocity ##############################################
        row = 0
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.obs[j, 6, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('vx (m/s)')
        axs[row, col].set_ylim([-2.0, 2.0])
        row = 1
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.obs[j, 7, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('vy (m/s)')
        axs[row, col].set_ylim([-2.0, 2.0])
        row = 2
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.obs[j, 8, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('vz (m/s)')
        axs[row, col].set_ylim([-2.0, 2.0])

        #### RPY Rates #############################################
        row = 3
        for j in range(self.NUM_DRONES):
            rdot = np.hstack([0, (self.obs[j, 3, 1:] - self.obs[j, 3, 0:-1]) * self.LOGGING_FREQ_HZ ])
            axs[row, col].plot(t, rdot, label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('rdot (rad/s)')
        axs[row, col].set_ylim([-1.0, 1.0])
        row = 4
        for j in range(self.NUM_DRONES):
            pdot = np.hstack([0, (self.obs[j, 4, 1:] - self.obs[j, 4, 0:-1]) * self.LOGGING_FREQ_HZ ])
            axs[row, col].plot(t, pdot, label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('pdot (rad/s)')
        axs[row, col].set_ylim([-1.0, 1.0])
        row = 5
        for j in range(self.NUM_DRONES):
            ydot = np.hstack([0, (self.obs[j, 5, 1:] - self.obs[j, 5, 0:-1]) * self.LOGGING_FREQ_HZ ])
            axs[row, col].plot(t, ydot, label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        axs[row, col].set_ylabel('ydot (rad/s)')
        axs[row, col].set_ylim([-3.14, 3.14])

        ### This IF converts RPM into PWM for all drones ###########
        #### except drone_0 (only used in examples/compare.py) #####
        for j in range(self.NUM_DRONES):
            for i in range(12,16):
                if pwm and j > 0:
                    self.obs[j, i, :] = (self.obs[j, i, :] - 4070.3) / 0.2685

        #### RPMs ##################################################
        row = 6
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.actions[j, 0, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        if pwm:
            axs[row, col].set_ylabel('PWM0')
        else:
            axs[row, col].set_ylabel('ROLL')
            axs[row, col].set_ylim([-1.0, 1.0])
        row = 7
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.actions[j, 1, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        if pwm:
            axs[row, col].set_ylabel('PWM1')
        else:
            axs[row, col].set_ylabel('PITCH')
            axs[row, col].set_ylim([-1.0, 1.0])
        row = 8
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.actions[j, 2, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        if pwm:
            axs[row, col].set_ylabel('PWM2')
        else:
            axs[row, col].set_ylabel('YAW')
            axs[row, col].set_ylim([-1.0, 1.0])
        row = 9
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.actions[j, 3, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time')
        if pwm:
            axs[row, col].set_ylabel('PWM3')
        else:
            axs[row, col].set_ylabel('THRUST')
            axs[row, col].set_ylim([-1.0, 1.0])

        #### Drawing options #######################################
        for i in range (10):
            for j in range (2):
                axs[i, j].grid(True)
                axs[i, j].legend(loc='upper right',
                         frameon=True
                         )
        fig.subplots_adjust(left=0.06,
                            bottom=0.05,
                            right=0.99,
                            top=0.98,
                            wspace=0.15,
                            hspace=0.0
                            )

        if show:
            plt.show()
