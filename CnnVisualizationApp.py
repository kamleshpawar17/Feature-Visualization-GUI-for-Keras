"""
File name: CnnVisualizationApp.py
Author: Kamlesh Pawar
Affiliation: Monash University, Australia
Date created: 11 Jan 2019
Date last modified: 11 Jan 2019
Python Version: 2.7.15
"""

import os
import tkFileDialog
import tkMessageBox
from Tkinter import *

import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageTk
from PIL.Image import open
from keras.models import load_model, Model
from keras.utils import plot_model


def create_menu(top_info, top_menu, value_var):
    if isinstance(top_info, dict):
        for key in top_info.keys():
            menu = Menu(top_menu)
            top_menu.add_cascade(label=key, menu=menu)
            create_menu(top_info[key], menu, value_var)
        return
    else:
        for value in top_info:
            top_menu.add_radiobutton(label=value, variable=value_var, value=value)
        return


class FeatVis:
    def __init__(self, name="CNN Feature Visualization"):
        self.width = 14
        self.padx = 8
        self.pady = 10
        self.name = name
        self.modelName = 0
        self.imageName = 0
        self.root = Toplevel()
        self.root.geometry('680x350')
        self.root.title(self.name)
        self.tkvar_layer = StringVar(self.root)
        self.choices_layers = ['None']
        self.tkvar_layer.set(self.choices_layers[0])
        intchoices = [l for l in range(1, 12)]
        self.tkvar_r = IntVar(self.root)
        self.tkvar_c = IntVar(self.root)
        self.tkvar_r.set(6)
        self.tkvar_c.set(8)
        self.imglogo_obj = open('/Users/kpaw0001/PycharmProjects/CnnVisualizationApp/FeatVis.png')
        self.tkimage = ImageTk.PhotoImage(self.imglogo_obj)
        self.tkvar_cmap = StringVar(self.root)
        cmapchoices = {'Perceptually Uniform Sequential': ['viridis', 'plasma', 'inferno', 'magma', 'cividis'],
                       'Sequential': ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds', 'YlOrBr', 'YlOrRd',
                                      'OrRd', 'PuRd', 'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn',
                                      'YlGn'],
                       'Sequential (2)': ['binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink', 'spring',
                                          'summer', 'autumn', 'winter', 'cool', 'Wistia', 'hot', 'afmhot', 'gist_heat',
                                          'copper'],
                       'Diverging': ['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu', 'RdYlGn', 'Spectral',
                                     'coolwarm', 'bwr', 'seismic'],
                       'Qualitative': ['Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2', 'Set1', 'Set2', 'Set3',
                                       'tab10', 'tab20', 'tab20b', 'tab20c'],
                       'Miscellaneous': ['flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern', 'gnuplot',
                                         'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'hsv', 'gist_rainbow', 'rainbow',
                                         'jet', 'nipy_spectral', 'gist_ncar']}
        self.tkvar_cmap.set('viridis')
        self.saveFeatPath = 0

        self.l_model = Label(self.root, text="Select Keras Model File", font=('Arial', 16))
        self.l_model.grid(row=1, column=0, sticky=W, padx=self.padx, pady=self.pady)
        self.l_layer = Label(self.root, text="Choose a Layer", font=('Arial', 16))
        self.l_layer.grid(row=3, column=0, sticky=W, padx=self.padx, pady=self.pady)
        self.l_row = Label(self.root, text="No. of Rows to Display", font=('Arial', 14))
        self.l_row.grid(row=1, column=2, sticky=W, padx=self.padx, pady=self.pady)
        self.l_col = Label(self.root, text="No. of Columns to Display", font=('Arial', 14))
        self.l_col.grid(row=2, column=2, sticky=W, padx=self.padx, pady=self.pady)
        self.l_cmap = Label(self.root, text="Display colormap", font=('Arial', 14))
        self.l_cmap.grid(row=3, column=2, sticky=W, padx=self.padx, pady=self.pady)
        self.l_logo = Label(self.root, image=self.tkimage)
        self.l_logo.grid(row=5, column=2, columnspan=2, rowspan=3, sticky=EW, padx=self.padx, pady=self.pady)

        self.b_browse = Button(self.root, text="Browse", command=self.askopenModel, width=self.width)
        self.b_browse.grid(row=1, column=1, sticky=W, padx=self.padx, pady=self.pady)
        self.b_parse = Button(self.root, text="Parse Model", command=self.parse_model, width=self.width)
        self.b_parse.grid(row=2, column=1, sticky=W, padx=self.padx, pady=self.pady)
        self.b_inpimg = Button(self.root, text="Input Image", command=self.askopenImage, width=self.width)
        self.b_inpimg.grid(row=4, column=1, sticky=W, padx=self.padx, pady=self.pady)
        self.b_disp = Button(self.root, text="Display", command=self.display, width=self.width, state=DISABLED)
        self.b_disp.grid(row=4, column=2, sticky=W, padx=self.padx, pady=self.pady)
        self.b_closedisp = Button(self.root, text="Close All Display", command=self.close_all_fig, width=self.width)
        self.b_closedisp.grid(row=4, column=3, sticky=W, padx=self.padx, pady=self.pady)
        self.b_compfeat = Button(self.root, text="Compute Features", command=self.compute_feature, width=self.width,
                                 state=DISABLED)
        self.b_compfeat.grid(row=5, column=1, sticky=W, padx=self.padx, pady=self.pady)
        self.b_save = Button(self.root, text="Save Features", command=self.save_feature, width=self.width,
                             state=DISABLED)
        self.b_save.grid(row=6, column=1, sticky=W, padx=self.padx, pady=self.pady)

        self.o_layer = OptionMenu(self.root, self.tkvar_layer, *self.choices_layers)
        self.o_layer.grid(row=3, column=1, sticky=EW, padx=self.padx, pady=self.pady)
        self.o_row = OptionMenu(self.root, self.tkvar_r, *intchoices)
        self.o_row.grid(row=1, column=3, sticky=EW, padx=self.padx, pady=self.pady)
        self.o_col = OptionMenu(self.root, self.tkvar_c, *intchoices)
        self.o_col.grid(row=2, column=3, sticky=EW, padx=self.padx, pady=self.pady)

        self.o_cmap = Menubutton(self.root, textvariable=self.tkvar_cmap, indicatoron=True)
        topMenu = Menu(self.o_cmap, tearoff=False)
        self.o_cmap.configure(menu=topMenu)
        self.o_cmap.grid(row=3, column=3, sticky=EW, padx=self.padx, pady=self.pady)
        create_menu(cmapchoices, topMenu, self.tkvar_cmap)

    def parse_model(self):
        if self.modelName:
            self.model = load_model(self.modelName)
            plot_model(self.model, './model_featvis.png')
            self.choices_layers = [l.name for l in self.model.layers]
            self.o_layer = OptionMenu(self.root, self.tkvar_layer, *self.choices_layers,
                                      command=self.enabledisable_with_layer)
            self.o_layer.grid(row=3, column=1, sticky=EW, padx=self.padx, pady=self.pady)
            tkMessageBox.showinfo("Message",
                                  "model parsed and model plot is saved as model_featvis.png in the current\
                                   working directory")
        else:
            tkMessageBox.showinfo("Message", "Please select a Model file first")

    def askopenModel(self):
        self.modelName = tkFileDialog.askopenfilename(
            filetypes=(("NifTi files", "*.h5"), ("compressed NiFti files", "*.hdf5")))

    def save_feature(self):
        self.saveFeatPath = tkFileDialog.askdirectory()
        if self.saveFeatPath != 0 and self.tkvar_layer.get() != str('None'):
            np.save(os.path.join(self.saveFeatPath, self.tkvar_layer.get()), self.featout)

    def enabledisable_with_layer(self, *args):
        self.b_save.config(state=DISABLED)
        self.b_compfeat.config(state=NORMAL)

    def imageReader(self):
        img_obj = open(self.imageName)
        self.inputImage = np.array(img_obj)

    def askopenImage(self):
        self.imageName = tkFileDialog.askopenfilename(filetypes=(
            ("JPEG files", "*.jpg"), ("JPEG files", "*.jpeg"), ("PNG files", "*.png"), ("TIFF files", "*.tiff"),
            ("TIFF files", "*.tif"), ("BMP files", "*.bmp")))
        if not self.imageName:
            return
        else:
            self.imageReader()

    def compute_feature(self):
        if self.imageName:
            model1 = Model(inputs=self.model.input, outputs=self.model.get_layer(self.tkvar_layer.get()).output)
            x = np.expand_dims(self.inputImage, 0)
            self.featout = np.squeeze(model1.predict(x))
            self.b_save.config(state=NORMAL)
            self.b_disp.config(state=NORMAL)
        else:
            tkMessageBox.showinfo("Message", "Please select input image first")

    def display(self):
        nFeatures = self.featout.shape[-1]
        k = 0
        nfig = (np.ceil(np.float(nFeatures) / np.float(self.tkvar_r.get() * self.tkvar_c.get()))).astype('int')
        for n in range(nfig):
            figname = self.name + ' Feature Maps - ' + str(n)
            fig0 = plt.figure(tight_layout=True)
            fig0.suptitle(figname, y=0.02)
            for m in range(self.tkvar_r.get() * self.tkvar_c.get()):
                if k < nFeatures:
                    plt.subplot(self.tkvar_r.get(), self.tkvar_c.get(), m + 1)
                    fig = plt.imshow(self.featout[:, :, k], cmap=self.tkvar_cmap.get())
                    fig.axes.get_xaxis().set_visible(False)
                    fig.axes.get_yaxis().set_visible(False)
                    k += 1
                    plt.title(str(k))
            fig0.subplots_adjust(wspace=0, hspace=0)
        figname = self.name + ' - Input Image'
        fig0 = plt.figure(tight_layout=True)
        fig0.suptitle(figname, y=0.02)
        plt.imshow(np.squeeze(self.inputImage), cmap=self.tkvar_cmap.get())
        plt.show()

    def close_all_fig(selfself):
        plt.close('all')

    def __call__(self, *args, **kwargs):
        self.root.mainloop()
