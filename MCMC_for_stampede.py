#!/usr/bin/env python3 
## job name
#SBATCH --job-name=run_MCMC
##SBATCH --partition=skx-dev    # SKX node: 48 cores, 4 GB per core, 192 GB total
##SBATCH --partition=skx-normal    # SKX node: 48 cores, 4 GB per core, 192 GB total
#SBATCH --partition=normal    ## KNL node: 64 cores x 2 FP threads, 1.6 GB per core, 96 GB total
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1    ## MPI tasks per node
#SBATCH --cpus-per-task=1    ## OpenMP threads per MPI task
#SBATCH --time=48:00:00
#SBATCH --output=MCMC_job_%j.txt                                                                                   
#SBATCH --mail-user=agraus@utexas.edu
#SBATCH --mail-type=fail
#SBATCH --mail-type=end
#SBATCH --account=TG-AST140080

#The purpose of this program is to take what I've learned from the MCMC
#notebooks and put it in a program that will run on a supercomputer
#
#The one modification I need to make is to add in a loop over all folders
#and all isochrones and then save the maximum likelihood over all isochrones
#
#

import os, re, emcee
import numpy as np

def log_prior(theta):
    #This is the prior function, for now it's very simple just a uniform prior
    #on the reddening and distance modulus
    #
    #Remember this is log so in log a prior is 0.0 when it's true
    #and -inf when it's false

    dist_mod, reddening = theta
    if 0.0 < reddening < 1.0 and 12.0 < dist_mod < 16.0:
        return 0.0
    return -np.inf

#In this case x and y would be the star color (x)  and star (y)

def log_probability(theta, obs_cmd_color, obs_cmd_mag, isochrone_file = None):
    #This version of the function will take in a arbitrary file and 
    #use that, hopefully putting it in as an option doesn't interfere with emcee
    #
    #Okay that seems to work with the files (8/25)
    #
    #Input:
    # theta - this is the prior which emcee will use at the variables to marginalize over
    # obs_cmd_color - the observed CMD color from Hugs photometry
    # obs_cmd_mag - the observed CMD magnitude from Hugs photometry
    # isochrone_file - the theoretical isochrone file from basti (you could probably put in whatever you want but would have to modify the code a bit
    #
    #Returns:
    # lp+likelihood - log of the prior plus the log likelihood

    import sys
    import numpy as np
    
    dist_mod, reddening = theta #emcee assumes your prior parameters are in theta
    lp = log_prior(theta)
    
    #Now load in a given isochrone:
    if isochrone_file == None:
        print('please define an isochrone to use')
        sys.exit()
    else:
        f_basti = np.loadtxt(isochrone_file)
        basti_f275w = f_basti[:,6]
        basti_f336w = f_basti[:,7]
        basti_f438w = f_basti[:,9]
        basti_f606w = f_basti[:,12]
        basti_f814w = f_basti[:,15]

    basti_color = (basti_f606w-basti_f814w)+reddening #Here is where our reddening comes into play
    basti_mag = basti_f606w+dist_mod #Here is where the distance modulus comes into play
    
    chi_color_list, chi_mag_list = [],[]

    for ii in range(len(F606)-1):

        obs_cmd_mag = F606[ii]
        obs_cmd_color = F606[ii]-F814[ii]

        #I'm not including the observed errors quite yet, because there's no real reason to yet
        #obs_cmd_mag_err = F606_RMS[ii]
        #obs_cmd_color_err = np.sqrt(F606_RMS[ii]**2.0+F814_RMS[ii]**2.0)

        cmd_dist = np.sqrt((basti_color-(obs_cmd_color))**2.0+(basti_mag-obs_cmd_mag)**2.0) #distance between observed point and isochrone points
        cmd_arg = np.argmin(cmd_dist) #find the point on the 
        chi_mag = (obs_cmd_mag - basti_mag[cmd_arg])
        chi_color = (obs_cmd_color - basti_color[cmd_arg])

        chi_color_list.append(chi_color)
        chi_mag_list.append(chi_mag)

    #assert shape(chi_mag_list)==shape(np.array(chi_mag_list)**2.0)

    Likelihood = -np.sum(np.array(chi_mag_list)**2+np.array(chi_color_list)**2) #Likelihood which is the sum of all the distances between the observed data and it's closest point on the isochrone

    #This returns -np.inf if outside the prior parameters
    if not np.isfinite(lp):
        return -np.inf
    
    return lp + Likelihood


#Now that the functions are in all I need to do is load the Hugs photometry and then loop over the isochrones

print('loading HUGS photometry')

f = np.loadtxt('./Hugs_photometry/hlsp_hugs_hst_wfc3-uvis-acs-wfc_ngc2808_multi_v1_catalog-meth1.txt',
              dtype=object)

F275 = np.array(f[:,2],dtype=float)
F336 = np.array(f[:,8],dtype=float)
F438 = np.array(f[:,14],dtype=float)
F606 = np.array(f[:,20],dtype=float)
F814 = np.array(f[:,26],dtype=float)

F275_RMS = np.array(f[:,3],dtype=float)
F336_RMS = np.array(f[:,9],dtype=float)
F438_RMS = np.array(f[:,15],dtype=float)
F606_RMS = np.array(f[:,21],dtype=float)
F814_RMS = np.array(f[:,27],dtype=float)

membership_prob = np.array(f[:,32],dtype=float) 

#now I need to mask out the weird values (photometry set to -99.99999) and stars with
#large errors F275W > 0.03, F336W > 0.03, F438W > 0.02
obs_mask = (F275>-99.0)&(F336>-99.0)&(F438>-99.0)&(F606>-99.0)&(F814>-99.0)&(F275_RMS<0.03)&(F336_RMS<0.03)&(F438_RMS<0.02)
obs_mask_no_err = (F275>-99.0)&(F336>-99.0)&(F438>-99.0)&(F606>-99.0)&(F814>-99.0)

F275_orig = F275.copy()
F336_orig = F336.copy()
F438_orig = F438.copy()
F606_orig = F606.copy()
F814_orig = F814.copy()

F275 = F275[obs_mask]
F336 = F336[obs_mask]
F438 = F438[obs_mask]
F606 = F606[obs_mask]
F814 = F814[obs_mask]

#Now we're going to loop over every isochrone

max_likelihood = -np.inf
basti_file_final = None

for basti_folder in os.listdir('./Basti_isochrones/'): 
    if basti_folder.startswith('FEH'):
        for basti_file in os.listdir('./Basti_isochrones/'+str(basti_folder)):
            print('running MCMC for {} ...'.format(basti_file))

            basti_loc = './Basti_isochrones/'+str(basti_folder)+'/'+str(basti_file)

            obs_cmd_color, obs_cmd_mag = F606-F814,F606

            #run the MCMC for this isochrone
            pos = np.array([15.,0.5]) + 1.0e-3*np.random.randn(5, 2) #initial guess for the MCMC
            nwalkers, ndim = 5, 2 #ndim is the number of prior parameters, nwalkers is the number if independent tasks
            sampler = emcee.EnsembleSampler(nwalkers,ndim, log_probability, args=(obs_cmd_color, obs_cmd_mag,basti_loc))

            sampler.run_mcmc(pos, 500, progress=False)

            #Now I need the best fit parameters
            samples = sampler.get_chain() #This is the full chain of the MCMC best value at the end
            log_prob_chain = sampler.get_log_prob()
            best_fit_modulus = np.median(samples[-1][:,0])
            best_fit_reddening = np.median(samples[-1][:,1])
            best_fit_log_likelihood = np.mean(log_prob_chain[-1]) #end state likelihoods IS MEDIAN A GOOD IDEA??
            #Samples[-1] gives the end state of every walker (it's a 3d array) and then each column is for
            #a different prior variable

            if best_fit_log_likelihood > max_likelihood:
                max_likelihood = best_fit_log_likelihood
                basti_file_final = basti_file

split_final = re.split('z|y|O',basti_file)
age_final = float(split_final[0])/1000.0 #Age of best fit isochrone Gyr
Z_final = float('0.'+split_final[1]) #Z of best fit isochrone
print('best fit file is:')
print(basti_file)
print('best fit age: {}'.format(age_test))
print('best fit Z: {}'.format(Z_test))
