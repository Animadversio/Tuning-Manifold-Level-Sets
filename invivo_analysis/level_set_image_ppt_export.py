
from core.utils import *
from core.utils.plot_utils import *
from core.utils.pptx_utils import *
from invivo_analysis import *
#%%
def layout_proto_evol_slide(slide, title_text, protopath, evol_figpath, manif_figpath,
                            contour_figpath, levelset_imgpath):
    """Template script for layouting 3 figures
    Layout a slide with a title and two figures.
    """
    tf = slide.shapes.title
    tf.text = title_text
    tf.text_frame._set_font("Candara", 28, False, False)
    for k, v in {'height': 1.041, 'width': 6.859, 'top': 0.0, 'left': 0.038}.items():
        setattr(tf, k, Inches(v))
    pic0 = slide.shapes.add_picture(protopath, Inches(0.0), Inches(0.0), )
    pic1 = slide.shapes.add_picture(evol_figpath, Inches(0.0), Inches(0.0), )
    pic2 = slide.shapes.add_picture(manif_figpath, Inches(0.0), Inches(0.0), )
    pic3 = slide.shapes.add_picture(contour_figpath, Inches(0.0), Inches(0.0), )
    pic4 = slide.shapes.add_picture(levelset_imgpath, Inches(0.0), Inches(0.0), )
    for k, v in {'height': 1.69, 'width': 1.69, 'top': 2.905, 'left': 0.000}.items():
        setattr(pic0, k, Inches(v))
    for k, v in {'height': 3.238, 'width': 3.533, 'top': 1.479, 'left': 1.681}.items():
        setattr(pic1, k, Inches(v))
    for k, v in {'height': 3.238, 'width': 3.238, 'top': 1.479, 'left': 5.14}.items():
        setattr(pic2, k, Inches(v))
    for k, v in {'height': 4.595, 'width': 5.13, 'top': 0.0, 'left': 8.203}.items():
        setattr(pic3, k, Inches(v))
    for k, v in {'height': 2.675, 'width': 13.333, 'top': 4.825, 'left': 0.0}.items():
        setattr(pic4, k, Inches(v))

#%%
lvlset_figdir = r"E:\OneDrive - Harvard University\Manifold_sphere_interp\summary\topology"
lvlset_imgdir = r"E:\OneDrive - Harvard University\Manifold_sphere_interp\levelset_images"
sumdir = r"E:\OneDrive - Harvard University\Manifold_NeuralRegress\summary"
outdir = r"E:\OneDrive - Harvard University\Manifold_sphere_interp\summary"
prs = Presentation()
# 16:9 wide screen layout
leveli, segi = 10, 0
prs.slide_width = Inches(13.33333)
prs.slide_height = Inches(7.5)
blank_slide_layout = prs.slide_layouts[5]
for Animal in ["Alfa", "Beto"]: #
    for Expi in tqdm(range(1, ExpNum[Animal] + 1)):
        meta = load_meta(Animal, Expi)
        title_str = meta.expstr
        protopath = join(sumdir, "proto", f"{Animal}_Exp{Expi:02d}_manif_proto.png")
        evolfigpath = join(sumdir, "classic_figs", f"{Animal}_Exp{Expi:02d}_Evolution.png")
        maniffigpath = join(sumdir, "classic_figs", f"{Animal}_Exp{Expi:02d}_Manifold.png")
        # ccfigpath = join(corrfeat_figdir, f"{Animal}_Exp{Expi:02d}_summary.png")
        lvlsetfigpath = join(lvlset_figdir, f"{Animal}_Exp{Expi:02d}_levelsets_all.png")
        lvlsetimgpath = join(lvlset_imgdir, f"{Animal}_Exp{Expi:02d}_contour_imgs_{leveli}-{segi}.jpg")
        lvlsetimgpath_rfmsk = join(lvlset_imgdir, f"{Animal}_Exp{Expi:02d}_contour_imgs_{leveli}-{segi}_rfmsk.jpg")
        slide = prs.slides.add_slide(blank_slide_layout)
        layout_proto_evol_slide(slide, title_str, protopath, evolfigpath, maniffigpath,
                            lvlsetfigpath, lvlsetimgpath_rfmsk)

prs.save(join(outdir, f"Levelset_images_pptx_export_{leveli}-{segi}.pptx"))
#%%
view_layout_params(join(outdir, "Levelset_Template.pptx"),0);

