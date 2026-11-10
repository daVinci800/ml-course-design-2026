"""CPU-only anatomical geometry control. Never reads test images or writes a submission."""
import time


def run():
    import cv2
    import numpy as np
    from final_round import ROOT, REPORT, inputs, read, write, score, crop

    cv2.setNumThreads(2)
    started = time.monotonic()
    manifest, records, _, baseline = inputs()
    features, targets = [], []
    for r in records:
        # Brightness gives an imperfect independent proxy for the optic disc.
        image = cv2.imread(r['path'])
        image, _ = crop(image, r['fov_center'], r['fov_side'])
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
        green = image[:, :, 1].astype(np.float32) / 255
        blurred = cv2.GaussianBlur(green, (0, 0), 7)
        yy, xx = np.mgrid[:256, :256]
        interior = (xx-127.5)**2 + (yy-127.5)**2 < 108**2
        masked = np.where(interior, blurred, -1)
        disc_y, disc_x = np.unravel_index(masked.argmax(), masked.shape)
        dx, dy = (disc_x-127.5)/256, (disc_y-127.5)/256
        side = r['fov_side']
        means = image[interior].mean(axis=0) / 255
        features.append([dx, dy, dx*dy, dx*dx, dy*dy, np.sign(dx),
                         r['width']/side, r['height']/side,
                         r['fov_center'][0]/r['width'], r['fov_center'][1]/r['height'],
                         *means, green[interior].std(), float(masked.max())])
        targets.append((np.asarray(r['xy'])-r['fov_center'])/side)
    x, y = np.array(features), np.array(targets)
    prediction = {}
    for fold in range(5):
        train = np.array([r['fold'] != fold for r in records])
        valid = ~train
        mean, std = x[train].mean(0), np.maximum(x[train].std(0), 1e-6)
        train_x = np.column_stack([np.ones(train.sum()), (x[train]-mean)/std])
        valid_x = np.column_stack([np.ones(valid.sum()), (x[valid]-mean)/std])
        penalty = np.eye(train_x.shape[1])*10
        penalty[0, 0] = 0
        coefficients = np.linalg.solve(train_x.T@train_x+penalty, train_x.T@y[train])
        for r, normalized in zip(np.array(records, dtype=object)[valid], valid_x@coefficients):
            point = normalized*r['fov_side']+r['fov_center']
            point = np.clip(point, [0, 0], [r['width']-1, r['height']-1])
            prediction[r['id']] = dict(xy=point.tolist())
    # This is a probe, deliberately kept separate from the pre-registered GPU queue.
    blends = {}
    for amount in (.01, .025, .05):
        p = {r['id']:dict(xy=((1-amount)*np.asarray(baseline[r['id']]['xy'])+
                             amount*np.asarray(prediction[r['id']]['xy'])).tolist()) for r in records}
        blends[str(amount)] = score(records, p)
    report = dict(standalone_rmse=score(records, prediction), baseline_rmse=score(records, baseline),
                  blend_rmse=blends, ridge_alpha=10, folds=5, training_images_only=True,
                  feature_scaling_fitted_per_training_fold=True, test_images_read=0,
                  wall_seconds=time.monotonic()-started,
                  decision='Diagnostic only; no additional GPU stage and no modification to final selection.',
                  limitation='Brightness is only an optic-disc proxy; anatomy alone lacks fine retinal localization.')
    write(REPORT/'geometry_probe.json', report)
    print(report, flush=True)


if __name__ == '__main__':
    run()
