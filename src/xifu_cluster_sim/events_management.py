from glob import glob
import os
from astropy.io import fits
import subprocess
from astropy import units as units
from .xifu_config import XIFU_Config

def merge_evt_files_recursive(output_file,
                            pattern,
                            clobber=False,
                            file_list=None,
                            depth=0):
    '''
    Merges different event files matching a given pattern recursively 
    
    Parameters:
        output_file (str): name of the final event list to write
        pattern (str): pattern to find the event files to merge
        clobber (bool): standard FITS clobber option
        file_list (list): list of files to merge (for recursive calls)
        depth (int): depth of recursive call
    '''        
    if file_list==None:
        file_list = glob(pattern)
        
    if os.path.exists(output_file):
        if not clobber : 
            raise RuntimeError("File %s already exists!" % output_file)

    nb_files = len(file_list)
    if nb_files==1:
        os.system("cp -r " + file_list[0] + " " + output_file)
    else:
        if nb_files>3:
            outfile1 = os.path.dirname(output_file)+'/tmp_evt_%d_0.fits' % depth
            merge_evt_files_recursive(outfile1,'',clobber=True,file_list=file_list[:int(nb_files/2)],depth=depth+1)
            outfile2 = os.path.dirname(output_file)+'/tmp_evt_%d_1.fits' % depth
            merge_evt_files_recursive(outfile2,'',clobber=True,file_list=file_list[int(nb_files/2):],depth=depth+1)
            print('Merging',outfile1,outfile2)
            os.system("fmerge \"%s %s\" %s \" \" clobber=yes" % (outfile1,outfile2,output_file))
        elif nb_files>0:
            print("Merging",file_list[0],file_list[1])
            os.system("fmerge \"%s %s\" %s \" \" clobber=yes" % (file_list[0],file_list[1],output_file))
            for evt_file in file_list[2:]:
                print('Merging',output_file,evt_file)
                os.system("fmerge \"%s %s\" %s \" \" clobber=yes" % (output_file,evt_file,output_file))
        if depth==0:
            for tmp_file in glob(os.path.dirname(output_file)+"/tmp_evt_*_*.fits"):
                os.remove(tmp_file)
                
    if os.path.exists(output_file):
        # Copy std GTI extension into final file
        hdu_evt = fits.open(output_file)
        hdu_evt0 = fits.open(file_list[0])
        hdu_evt.append(hdu_evt0['STDGTI'])
        hdu_evt.writeto(output_file, overwrite=True)
        hdu_evt.close()

def create_count_map(final_evt_file,
                    image_file,
                    ra =0.,
                    dec = 0.,
                    count_map_shape = (58,58),
                    xifu_config = XIFU_Config()):
    
    pixsize_degree = xifu_config.pixsize_arcsec.to(units.degree).value

    subprocess.check_call(["imgev",
                            "EvtFile="+final_evt_file,
                            "Image="+image_file,
                            "NAXIS1={}".format(count_map_shape[0]),
                            "NAXIS2={}".format(count_map_shape[1]),
                            "CRVAL1={}".format(ra),
                            "CRVAL2={}".format(dec),
                           "CRPIX1={}".format(int(count_map_shape[0]/2)),
                           "CRPIX2={}".format(int(count_map_shape[1]/2)),
                           "CDELT1=-%.15f" %(pixsize_degree),
                           "CDELT2=%.15f" %(pixsize_degree),
                           "CoordinateSystem=0",
                           "Projection=TAN",
                           "CUNIT1=deg",
                           "CUNIT2=deg"])
