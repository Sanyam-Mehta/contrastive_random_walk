# Code Taken From: https://github.com/laughtervv/DepthAwareCNN/blob/master/utils/visualizer.py

import ntpath
import os
from io import BytesIO  # Python 3.x

from contrastive_random_walk.viz import html, util
from PIL import Image


class Visualizer:
    def __init__(self, 
    tf_log=True,
    use_html=True,
    win_size=256,
    name="contrastive_random_walk_train",
    freq=100,
    ):
        # self.opt = opt
        # self.tf_log = config.VISUALIZE.TF_LOG
        # self.use_html = config.VISUALIZE.IS_TRAIN and config.VISUALIZE.USE_HTML
        # self.win_size = config.VISUALIZE.DISPLAY_WIN_SIZE
        # self.name = config.VISUALIZE.NAME
        # self.freq = config.VISUALIZE.VISUALIZE_FREQUENCY
        self.tf_log = tf_log
        self.use_html = use_html
        self.win_size = win_size
        self.name = name
        self.freq = freq
        self.output_dir = "data/checkpoint"
        self.viz_name = "exp1"
        
        if self.tf_log:
            import tensorflow as tf

            self.tf = tf
            self.tf.compat.v1.disable_eager_execution()
            self.log_dir = os.path.join(
                self.output_dir, self.viz_name, "logs"
            )
            self.writer = tf.compat.v1.summary.FileWriter(self.log_dir)

        if self.use_html:
            self.web_dir = os.path.join(self.output_dir, self.viz_name, "web")
            self.img_dir = os.path.join(self.web_dir, "images")
            print("create web directory %s..." % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, step):

        display_scores = False

        if self.tf_log:  # show images in tensorboard output
            img_summaries = []
            for label, image_numpy in visuals.items():
                # Write the image to a string
                # try:
                #     s = StringIO()
                # except:
                #     s = BytesIO()
                s = BytesIO()
                try:
                    Image.fromarray(image_numpy).save(s, format="jpeg")
                except:
                    breakpoint()

                # Create an Image object
                img_sum = self.tf.compat.v1.Summary.Image(
                    encoded_image_string=s.getvalue(),
                    height=image_numpy.shape[0],
                    width=image_numpy.shape[1],
                )
                # Create a Summary value
                img_summaries.append(
                    self.tf.compat.v1.Summary.Value(tag=label, image=img_sum)
                )

            # Create and write Summary
            summary = self.tf.compat.v1.Summary(value=img_summaries)
            self.writer.add_summary(summary, step)

        if self.use_html:  # save images to a html file
            for label, image_numpy in visuals.items():
                if isinstance(image_numpy, list):
                    for i in range(len(image_numpy)):
                        img_path = os.path.join(
                            self.img_dir, "step%.3d_%s_%d.jpg" % (step, label, i)
                        )
                        util.save_image(image_numpy[i], img_path)
                else:
                    img_path = os.path.join(
                        self.img_dir, "step%.3d_%s.jpg" % (step, label)
                    )
                    util.save_image(image_numpy, img_path)

            # update website
            webpage = html.HTML(self.web_dir, self.name, refresh=5000)
            for n in range(step, 0, -self.freq):
                # webpage.add_header('step [%d]' % n)
                # if dataset_image_list is not None:
                #    mission_name, img_name = dataset_image_list[n].parts[-2:]
                # else:
                #    mission_name, img_name = "", ""

                # if display_scores:
                #     webpage.add_header(f'mission_name: {mission_name} img name: {img_name}' )
                # else:
                #     webpage.add_header(f'mission_name: {mission_name} img name: {img_name}')
                ims = []
                txts = []
                links = []

                for label, image_numpy in visuals.items():
                    if isinstance(image_numpy, list):
                        for i in range(len(image_numpy)):
                            img_path = "step%.3d_%s_%d.jpg" % (n, label, i)
                            ims.append(img_path)
                            txts.append(label + str(i))
                            links.append(img_path)
                    else:
                        img_path = "step%.3d_%s.jpg" % (n, label)
                        ims.append(img_path)
                        txts.append(label)
                        links.append(img_path)
                if len(ims) < 10:
                    webpage.add_images(ims, txts, links, width=self.win_size)
                else:
                    num = int(round(len(ims) / 2.0))
                    webpage.add_images(
                        ims[:num], txts[:num], links[:num], width=self.win_size
                    )
                    webpage.add_images(
                        ims[num:], txts[num:], links[num:], width=self.win_size
                    )
            webpage.save()

    # errors: dictionary of error labels and values
    def plot_current_errors(self, errors, step):
        if self.tf_log:
            for tag, value in errors.items():
                summary = self.tf.compat.v1.Summary(
                    value=[self.tf.compat.v1.Summary.Value(tag=tag, simple_value=value)]
                )
                self.writer.add_summary(summary, step)

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t):
        message = "(epoch: %d, iters: %d, time: %.3f) " % (epoch, i, t)
        for k, v in errors.items():
            if v != 0:
                message += "%s: %.3f " % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write("%s\n" % message)

    # save image to the disk
    def save_images(self, webpage, visuals, image_path):
        image_dir = webpage.get_image_dir()
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            image_name = "%s_%s.jpg" % (name, label)
            save_path = os.path.join(image_dir, image_name)
            util.save_image(image_numpy, save_path)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=self.win_size)
