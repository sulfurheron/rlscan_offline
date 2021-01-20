import glob, os, shutil
import datetime
import pytz
import numpy as np
import pandas as pd
from io import BytesIO
from threading import Thread

from rlscan.policies.utils import pose_features_from_obs_acs_cont
from rlscan_offline.utils import scanner_frames_from_obs
from orb.scan_routine_stubs import scan_routine_stubs_pb2
from rlscan_train.utils.s3_helpers import SMSDownloader, S3Downloader
from rlscan_train.utils.pb_helpers import read_episode_proto_file, save_episode_proto_file
import time
import pickle
from PIL import Image



class DataReader:
    MACHINE_NAME = 'gapglt-orb191.ext.kindredai.net'
    SAVE_DIR_TEMP = "/media/dmytro/ExtraDrive1/Data/rlscan/offline/temp"
    SAVE_DIR = "/media/dmytro/ExtraDrive1/Data/rlscan/offline"
    # Event time is in GMT, NOT ET!!
    # SMS Bastion Credentials
    USERNAME = "dmytro"
    SSH_KEY_FILE = "/home/dmytro/.ssh/id_rsa_old"
    SMS_PW = "/home/dmytro/.ssh/sms_pw"
    STAGING_SMS_PW = "/home/dmytro/.ssh/staging_sms_pw"
    def __init__(self):
        self._interval_length = 1 # hours
        self.setup_connection()

    def setup_connection(self):
        self.interval_start = '2020-10-14 00:00:00'
        internal_orb = False
        self.pw_file = self.STAGING_SMS_PW if internal_orb else self.SMS_PW
        # print("pw_file", self.pw_file)
        self.sms = SMSDownloader(username=self.USERNAME, ssh_key_file=self.SSH_KEY_FILE, pw_file=self.pw_file)
        self.s3_download = S3Downloader(save_dir=self.SAVE_DIR_TEMP)
        np.set_printoptions(precision=3)

    def query_data_from_sms(self, interval):
        print("Pulling interval", interval)
        sms_data = self.sms.rlscan_episode_query(interval=interval, machine_name=self.MACHINE_NAME)
        phase_group_ids = list(sms_data.keys())
        urls = []
        for pg in phase_group_ids:
            # print(rows[pg]['scanner_id'], pg, rows[pg]['url'].replace('%2F', '/'))
            # TODO: Fix error and do away with replace.
            #  Sometimes SMS does weird shit. / is replaced with unicode %2F breaking SMS downloader.
            url = sms_data[pg]['url'].replace('%2F', '/')
            urls.append(url)
        print("Found {} episodes in the specified interval".format(len(urls)))
        return sms_data, urls

    def download_thread_body(self, new_urls):
        try:
            for i_url in range(0, len(new_urls), 100):
                tic = time.time()
                start_ind = i_url
                end_ind = min(i_url + 100, len(new_urls))
                slice = new_urls[start_ind: end_ind]
                self.s3_download.download(slice)
                print("Downloaded {} of {} files. Time taken: {}s".format(end_ind, len(new_urls), time.time() - tic))
        except Exception as e:
            print("Error downloading data", e)


    def pull_new_data_batch(self):
        total_steps = 0
        total_episodes = 0
        explored_fps = []
        int_start_str = self.interval_start
        current_dir_id = 0
        current_file_id = 0
        data_dict = {
            'image_file': [],
            'pos_features': [],
            'action': [],
            'phase_group_id': [],
            "timestep": [],
            "scanner_id": [],
            "ep_len": []
        }
        while True:
            #tentative_new_start = (datetime.datetime.now(pytz.timezone('GMT'))).strftime("%Y-%m-%d %H:%M:%S")
            int_start_dt = datetime.datetime.strptime(int_start_str, "%Y-%m-%d %H:%M:%S")
            int_end_dt = int_start_dt + datetime.timedelta(hours=self._interval_length)
            int_end_str = int_end_dt.strftime("%Y-%m-%d %H:%M:%S")
            interval = (int_start_str, int_end_str)
            print("interval", interval)
            int_start_str = int_end_str
            if int_start_dt >= datetime.datetime.now():
                break
            try:
                sms_data, new_urls = self.query_data_from_sms(interval)
            except Exception as e:
                print("Error querying data", e)
                self.sms = SMSDownloader(username=self.USERNAME, ssh_key_file=self.SSH_KEY_FILE, pw_file=self.pw_file)
                time.sleep(300)
                continue
            if not len(new_urls):
                continue
            # check the directory exists:
            if os.path.isdir(self.SAVE_DIR_TEMP):
                try:
                    shutil.rmtree(self.SAVE_DIR_TEMP)
                except OSError as e:
                    print("Error cleaning directory: %s : %s" % (self.SAVE_DIR_TEMP, e.strerror))
            os.mkdir(self.SAVE_DIR_TEMP)
            while True:
                th = Thread(target=self.download_thread_body, args=(new_urls,))
                th.start()
                th.join(timeout=600)
                if not th.is_alive():
                    break
                time.sleep(120)
                self.s3_download = S3Downloader(save_dir=self.SAVE_DIR_TEMP)


            # Create a list of downloaded files and save it locally as a data manifest
            filepaths = glob.glob(self.SAVE_DIR_TEMP + "/*/*.pb")
            print("{} files found in {}".format(len(filepaths), self.SAVE_DIR_TEMP))
            if len(filepaths):
                for fp in filepaths:
                    try:
                        with open(fp, "rb") as f:
                            episode_pb = scan_routine_stubs_pb2.Episode()
                            episode_pb.ParseFromString(f.read())
                            explored_fps.append(fp)
                    except Exception as e:
                        print("Failed to read episode file", e)
                        continue
                    if episode_pb.policy_class == "ScanMotionPolicy_PPO_LSTM":
                        action = np.zeros(2)
                        for i in range(len(episode_pb.observations) - 1):
                            frames, pos_features, action = self.episode_to_obs(episode_pb, action, i)
                            image_filename = "{}_{}_{}.pkl".format(
                                episode_pb.phase_group_id,
                                i,
                                current_file_id
                            )
                            image_path = os.path.join(self.SAVE_DIR, str(current_dir_id), image_filename)
                            data_dict['image_file'].append(image_path)
                            data_dict['pos_features'].append(pos_features)
                            data_dict['action'].append(action)
                            data_dict['phase_group_id'].append(episode_pb.phase_group_id)
                            data_dict['scanner_id'].append(sms_data[episode_pb.phase_group_id]['scanner_id'])
                            data_dict['ep_len'].append(len(episode_pb.observations) - 1)
                            current_dir = os.path.join(self.SAVE_DIR, str(current_dir_id))
                            if not os.path.isdir(current_dir):
                                os.mkdir(current_dir)
                            #with open(image_path, "wb") as f:
                            #    pickle.dump(frames, f)
                            jpgs = []
                            for frame in frames:
                                im = Image.fromarray(frame)
                                b = BytesIO()
                                im.save(b,  'jpeg')
                                jpgs.append(b)
                            with open(image_path, "wb") as f:
                                pickle.dump(jpgs, f)

                            #np.save(image_path, frames)
                            current_file_id += 1
                            if current_file_id > 1000:
                                current_dir_id += 1
                                current_file_id = 0
                        total_steps += len(episode_pb.observations)
                        total_episodes += 1
                        # print("updating scanner id", episode_pb.phase_group_id, sms_data[episode_pb.phase_group_id]['scanner_id'])
                        # scanner_ids[episode_pb.phase_group_id] = sms_data[episode_pb.phase_group_id]['scanner_id']
                with open(os.path.join(self.SAVE_DIR, "metadata.pkl"), "wb") as f:
                    pickle.dump(data_dict, f)

                print("total time steps {}, episodes {}".format(total_steps, total_episodes))

                return
                # if total_steps >= self.n_steps:
                #     self.interval_start = tentative_new_start
                #     return episodes, scanner_ids
                #break
            #time.sleep(300)

    def episode_to_obs(self, episode_pb, action, i):
        pos_features, _, state= pose_features_from_obs_acs_cont(episode_pb.observations[i], np.zeros(2), action, extract_state=True)
        scanner_frames = scanner_frames_from_obs(episode_pb.observations[i])
        action = episode_pb.actions[i].agent_action
        return scanner_frames, pos_features, np.array(action).astype('float32')

if __name__ == "__main__":
    d = DataReader()
    d.pull_new_data_batch()
