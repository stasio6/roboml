import os
import shutil
import wandb
api = wandb.Api(timeout=30)

SAVE_DIR = "data/obs_lift_plots"

def get_runs(exp_name, entity, project, run_ids, clean_dir=True):
    dir_to_save = f'{SAVE_DIR}/{exp_name}'
    if clean_dir:
        shutil.rmtree(dir_to_save, ignore_errors=True)
    for run_id in run_ids:
        run = api.run(f"{entity}/{project}/{run_id}")

        # save the metrics for the run to a csv file
        metrics_dataframe = run.history()
        os.makedirs(dir_to_save, exist_ok=True)
        metrics_dataframe.to_csv(f"{dir_to_save}/{run_id}.csv")
    print(f'{exp_name} downloaded.')

if __name__ == "__main__":

    ###############################################
    # ManiSkill
    ###############################################

    # get_runs("PickCube_SAC-state", 'rk-dev', "ManiSkill2-New", [
    #     'czmfj755', '69drhkap', 'shxjmblq', '0wjgpq1n', 'i56zf5v4'
    # ])
    # get_runs("PickCube_s2v", 'sstrzelecki', "roboml", [
    #     '1ph678o9', 'r5xqncgm', 'mll2b77n'
    # ])
    # get_runs("PickCube_AAC-SAC", 'sstrzelecki', "roboml", [
    #     'ua8cgwow', '4hhwvsb2', 'nqk73lsx'
    # ])

    # get_runs("StackCube_SAC-state", 'tmu', "ManiSkill2-dev", [
    #     'b69bp0z8','twjwri03','sa6ms39n'
    # ])
    # get_runs("StackCube_s2v", 'sstrzelecki', "roboml", [
    #     '0pu2x1sm', '8v44tqcv', '4fufz635'
    # ])
    # get_runs("StackCube_AAC-SAC", 'ycyao216', "Neurips-ms-rgbd-aac-sac-30Msteps", [
    #     '3aykx2xv','fverxgfr','g9ew3cb8'
    # ])

    # get_runs("PickSingleYCB_SAC-state", 'tmu', "ManiSkill2-dev", [
    #     '6jlxrwyl','icuvro8o','u6kqbqx4'
    # ])
    # get_runs("PickSingleYCB_s2v", 'sstrzelecki', "roboml", [
    #     '9366909o', 'q12bek6v', 'kwws972x'
    # ])
    # get_runs("PickSingleYCB_AAC-SAC", 'ycyao216', "Neurips-ms-rgbd-aac-sac-30Msteps", [
    #     'thl2q38b','8nhu1ak5','w0wpznjx'
    # ])

    # get_runs("TurnFaucet_SAC-state", 'tmu', "ManiSkill2-dev", [
    #     'u4tfkq09', '7brsyvdp', 'pukavffe'
    # ])
    # get_runs("TurnFaucet_s2v", 'sstrzelecki', "roboml", [
    #     'k9y4b1ma', 'gl6e69ot', 'aynuzcv2'
    # ])
    # get_runs("TurnFaucet_AAC-SAC", 'ycyao216', "Neurips-ms-rgbd-aac-sac-30Msteps", [
    #     'jpb0a2jv','1h1n8aeu','ypf781zi',
    # ])

    # get_runs("MoveBucket_SAC-state", 'tmu', "ManiSkill2-dev", [
    #     'kzf7is39','z0nxk1r8','75gbb3cb','eujarq74',
    # ])
    # get_runs("MoveBucket_s2v", 'sstrzelecki', "s2v-DAgger-baselines", [
    #     'yoat1jhv'
    # ])
    # get_runs("MoveBucket_s2v", 'sstrzelecki', "roboml", [
    #     'bzqqt6mv', 'dhmpe2ge'
    # ], clean_dir=False)
    # get_runs("MoveBucket_AAC-SAC", 'ycyao216', "Neurips-ms-rgbd-aac-sac-30Msteps", [
    #     '0qiw03lo', 'kla0gpdx', 'xzm69hdl'
    # ])

    # get_runs("OpenDrawer_SAC-state", 'tmu', "ManiSkill2-dev", [
    #     'yaz1keby','7kdcvrbu','q5uh2pb9'
    # ])
    # get_runs("OpenDrawer_s2v", 'sstrzelecki', "s2v-DAgger-baselines", [
    #     '6awsadh3',
    # ])
    # get_runs("OpenDrawer_s2v", 'sstrzelecki', "roboml", [
    #     '5xowqfbs', 'uflx16j6', 'gfebq85s'
    # ], clean_dir=False)
    # get_runs("OpenDrawer_AAC-SAC", 'ycyao216', "Neurips-ms-rgbd-aac-sac-30Msteps", [
    #     'tsitod3q', '1r99o68n', 'gg6clwxs'
    # ])
    # get_runs("OpenDrawer_AAC-SAC", 'sstrzelecki', "roboml", [
    #     's1zzrxth', 'anq0so9b'
    # ], clean_dir=False)

    # get_runs("PegInsertion_SAC-state", 'tmu', "ManiSkill2-dev", [
    #     'ubynkm9a','9mu4fwrv','uux12i8h'
    # ])
    # get_runs("PegInsertion_s2v", 'sstrzelecki', "roboml", [
    #     'rcnn622k', 'eff3vd25', 'zf2onvfs',
    # ])
    # get_runs("PegInsertion_AAC-SAC", 'ycyao216', "Neurips-ms-rgbd-aac-sac-30Msteps", [
    #     'hdaxfeze','fv67d84c','j96agsgt'
    # ])

    # get_runs("PickClutterYCB_SAC-state", 'rk-dev', "ManiSkill2-New", [
    #     '4v39ux6h','1z73mecf','swilb9lz', '9rpz3c0i', 'z2rli84p'
    # ])
    # get_runs("PickClutterYCB_s2v", 'sstrzelecki', "roboml", [
    #     'hdcxqq51', 'n7szu7go', 'yl6fj5as'
    # ])
    # get_runs("PickClutterYCB_AAC-SAC", 'sstrzelecki', "roboml", [
    #     'o2cp8ntu', 'kba3abk5', 'vhvszp4a'
    # ])

    # get_runs("OpenDoor_SAC-state", 'tmu', "ManiSkill2-dev", [
    #     'xl416bnf','dzz92qcn','m5qiiis1'
    # ])
    # get_runs("OpenDoor_s2v", 'sstrzelecki', "s2v-DAgger-baselines", [
    #     '0kf9e2wp',
    # ])

    ###############################################
    # DMControl
    ###############################################

    # get_runs("Acrobot-Swingup_SAC-state", 'ycyao216', "Neurips-DMC-S2V-tune", [
    #     '2ahouicv','7j9yvzqp', 'axy3dqd8',
    # ])
    # get_runs("Acrobot-Swingup_s2v", 'sstrzelecki', "roboml", [
    #     '3kqbqakz', 'jkqesmn3', 'w63lh7ln',
    # ])
    # get_runs("Acrobot-Swingup_AAC-SAC", 'ycyao216', "Neurips-dmc-rgb-aac_sac-rerun", [
    #     '42k8sz6j', 'ue1hpr3v', 'yzeqljmx',
    # ])

    # get_runs("Reacher-Hard_SAC-state", 'ycyao216', "Neurips-DMC-S2V-tune", [
    #     '3lhmngxr', '0m1zk42i', 'c88e0s7u',
    # ])
    # get_runs("Reacher-Hard_s2v", 'sstrzelecki', "roboml", [
    #     'ci9zuoxm', 'iszqn5ou', '1yftp45y'
    # ])
    # get_runs("Reacher-Hard_AAC-SAC", 'ycyao216', "Neurips-dmc-rgb-aac_sac-rerun", [
    #     'wykd3lvz', 's5sgcszv', 'v9dzym8x',
    # ])

    # get_runs("Swimmer-6_SAC-state", 'ycyao216', "Neurips-DMC-S2V-tune", [
    #     'gq7y3vm9', 'cxm5yr0s', 'xoevq84t',
    # ])
    # get_runs("Swimmer-6_s2v", 'sstrzelecki', "roboml", [
    #     'qzssg9q9', '0gbi9vxu', 'h04d3p4h',
    # ])
    # get_runs("Swimmer-6_AAC-SAC", 'ycyao216', "Neurips-dmc-rgb-aac_sac-rerun", [
    #     'txtvmmjw', 'fn84plmn', 'ym7du9oc',
    # ])

    # get_runs("Walker-Run_SAC-state", 'ycyao216', "Neurips-DMC-S2V-tune", [
    #     'vrasabzl', 'mt5d3cun', 'ltxjta1s',
    # ])
    # get_runs("Walker-Run_s2v", 'sstrzelecki', "roboml", [
    #     'x5uwtucx', 'ukepcwfb', 'cjzx0vh1',
    # ])
    # get_runs("Walker-Run_AAC-SAC", 'ycyao216', "Neurips-dmc-rgb-aac_sac-rerun", [
    #     'e5hyd49z', '3dv3pgl6', 'y9l737hz',
    # ])

    # get_runs("Hopper-Hop_SAC-state", 'ycyao216', "Neurips-DMC-S2V-tune", [
    #     't0mrvwa1', '6us2c4fk', 'fyqq6ybm',
    # ])
    # get_runs("Hopper-Hop_s2v", 'sstrzelecki', "roboml", [
    #     'n558xf3y', 'zou7yghu', 'dlb7rk2x',
    # ])
    # get_runs("Hopper-Hop_AAC-SAC", 'ycyao216', "Neurips-dmc-rgb-aac_sac-rerun", [
    #     'myfi8i4a', 'z2kka7rc', 'e3u6tj3n',
    # ])

    # get_runs("Humanoid-Walk_SAC-state", 'ycyao216', "Neurips-DMC-S2V-tune", [
    #     '06rp3op4', '6wfuw9gi', '87qtmicu',
    # ])
    # get_runs("Humanoid-Walk_s2v", 'sstrzelecki', "roboml", [
    #     'jn2y4ar5', '503spaie', 'q76ck13l',
    # ])
    # get_runs("Humanoid-Walk_AAC-SAC", 'ycyao216', "Neurips-dmc-rgb-aac_sac-rerun", [
    #     '5lcwry2p','r4dg13hg','so4fjpn4'
    # ])

    print("Done")