import bydoux_tools as bt
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
import os
from astropy.time import Time
#import datetime

import bydoux_tools as bt
from bydoux_tools import printc
# get today's date
import datetime
# import ius
from scipy.interpolate import UnivariateSpline as ius
from wpca import PCA
from scipy.special import erf

def simu_sp500(nkeeps=[10], timesteps=[45], lookback_scales=[7], method='slope', injection=30000, nyears=20, doplot=False):

    tickers0 = bt.get_snp500()
    tickers = np.array(tickers0)

    lookbacks = np.array(timesteps) * np.array(lookback_scales)


    today = bt.today()

    tbl = Table()
    tbl['nkeeps'] = nkeeps
    tbl['timestep'] = timesteps
    tbl['lookback_scale'] = lookback_scales
    tbl['method'] = method
    tbl['mean_growth_nothing'] = np.nan
    tbl['mean_growth_strategy'] = np.nan
    tbl['rms_nothing'] = np.nan
    tbl['rms_strategy'] = np.nan



    for nth_simu in range(len(nkeeps)):
        all_sp500 = bt.batch_quotes('S&P500', force=False)

        nkeep = nkeeps[nth_simu]
        timestep = timesteps[nth_simu]
        lookback_scale = lookback_scales[nth_simu]
        lookback = lookbacks[nth_simu]

        scenario_name = f'{timestep}days_{lookback}days_{nyears}years_{nkeep}stocks_{method}'

        t_decision = Time(datetime.datetime.now()).mjd - np.arange(0, 365 * nyears, timestep)

        tmax = all_sp500[tickers[0]]['mjd'][-1]


        growth_nothing = np.zeros(len(t_decision))
        growth_strategy = np.zeros(len(t_decision))

        t_decision = t_decision[np.argsort(t_decision)]


        sectors_ini = []
        # get all sectors:
        for ticker in tickers:
            try:
                info = bt.get_info(ticker)
                sect = info['sector']
                printc(f'{ticker}, sector {sect}')
                sectors_ini.append(info['sector'])
            except:
                sectors_ini.append('Unknown')

        sectors_ini = np.array(sectors_ini)

        dict_gains = {}
        for imc in range(len(t_decision)):
            tbls = []
            best_keys = []
            kept_keys = []
            best_sectors = []
            gains = []
            slopes = []
            sigmas = []
            gain_pred = []
            sectors = []

            tickers = np.array(tickers0)

            for iticker in range(len(tickers)):
                ticker = tickers[iticker]

                if ticker not in all_sp500.keys():
                    continue

                tbl1 = dict(all_sp500[ticker])
                if len(tbl1['date'])<100:
                    continue


                t1 = t_decision[imc] - lookback#[nth_simu]
                t2 = t_decision[imc]+timestep#[nth_simu]

                time_valid = (tbl1['mjd']>t1) & (tbl1['mjd']<t2)
                if np.sum(time_valid)<0.5*lookback:
                    continue

                for key_tbl in tbl1.keys():
                    tbl1[key_tbl] = tbl1[key_tbl][time_valid]

                dt = np.max(tbl1['mjd']) - tbl1['mjd']

                keep_for_fit = tbl1['mjd'] < t_decision[imc]

                if method == 'slope':
                    fit = np.polyfit(tbl1['mjd'][keep_for_fit] / 365.25, tbl1['log_close'][keep_for_fit],1)

                    rms = (tbl1['log_close'][keep_for_fit] - np.polyval(fit, tbl1['mjd'][keep_for_fit] / 365.25))

                    sigma = np.nanmean(np.abs(rms))

                    slopes.append(fit[0])
                    sigmas.append(sigma)

                elif method == 'mean':
                    slopes.append(tbl1['log_close'][keep_for_fit][-1] - np.nanmean(tbl1['log_close'][keep_for_fit][-1]))
                    sigmas.append(np.nanstd(tbl1['log_close'][keep_for_fit]))

                elif method == 'cheaper':
                    slopes.append( - tbl1['Close'][keep_for_fit][-1])
                    sigmas.append(np.nanstd(tbl1['log_close'][keep_for_fit]))


                last_iteration = (np.max(tbl1['mjd']) == t_decision[imc])

                if not last_iteration:
                    gain = tbl1['log_close'][-1]- tbl1['log_close'][keep_for_fit][-1]
                    gains.append(gain)
                else:
                    gains.append(np.nan)

                kept_keys.append(ticker)
                sectors.append(sectors_ini[iticker])

                outname = '/Users/eartigau/pydoux_data/quotes/' + ticker + '_' + today + '.csv'


            slopes = np.array(slopes)
            sigmas = np.array(sigmas)
            gains = np.array(gains)
            keys = np.array(kept_keys)
            sectors = np.array(sectors)


            # we keep only the best nkeep stocks
            ord = np.argsort(slopes)[::-1]

            # we reorder the arrays according to the order vector
            gains_keep = gains[ord]#[0:nkeep]
            slopes_keep = slopes[ord]#[0:nkeep]
            sigmas_keep = sigmas[ord]#[0:nkeep]
            keys_keep = keys[ord]#[0:nkeep]
            sectors_keep = sectors[ord]#[0:nkeep]

            if True:
                keep = np.array([False]*len(gains_keep))

                usector =  np.unique(sectors_keep)
                for sector in usector:
                    mask = sectors_keep == sector
                    if np.sum(mask)<3:
                        continue

                    tmp = keep[mask]
                    nmax = nkeep//4
                    tmp[0:nmax] = True
                    keep[mask] = tmp

                gains_keep = gains_keep[keep]
                slopes_keep = slopes_keep[keep]
                sigmas_keep = sigmas_keep[keep]
                keys_keep = keys_keep[keep]
                sectors_keep = sectors_keep[keep]

            prob_gain = np.zeros(len(gains_keep))+0.04
            for i in range(len(gains_keep)):
                if keys_keep[i] in dict_gains.keys():
                    prob_gain[i] = np.nanmean(dict_gains[keys_keep[i]])

            ord = np.argsort(keys_keep)
            gains_keep = gains_keep[0:nkeep]
            slopes_keep = slopes_keep[0:nkeep]
            sigmas_keep = sigmas_keep[0:nkeep]
            keys_keep = keys_keep[0:nkeep]
            sectors_keep = sectors_keep[0:nkeep]
            prob_gain = prob_gain[0:nkeep]


            #ww = erf((prob_gain-np.mean(prob_gain))/0.05)*0.5+0.5
            #ww[ww<0] = 0
            ##ww[ww>np.nanmedian(ww)] = 2*np.nanmedian(ww)
            #ww/=np.mean(ww)

            for key in keys_keep:
                if key not in dict_gains.keys():
                    dict_gains[key] = []
                dict_gains[key].append(gains_keep[np.where(keys_keep == key)[0][0]])
            

            os.system('clear')

            t_year = (t_decision[imc] - Time('2000-01-01').mjd) / 365.25 + 2000


            printc(f'Year : {t_year:.2f}')
            gain1 = np.nanmean(np.exp(gains)) #** (365 / dt2)
            printc(f'gain/{timestep:.0f} days doing nothing, {gain1:.2f}')
            gain2 = np.nanmean(np.exp(gains_keep)) #** (365 / dt2)
            printc(f'gain/{timestep:.0f} days with strategy, {gain2:.2f}')


            growth_nothing[imc] = gain1
            growth_strategy[imc] = gain2


            for i in range(len(gains_keep)):
                ptxt = keys_keep[i],np.exp(gains_keep[i]), np.exp(gains_keep[i] * (365 / timestep)), slopes_keep[i], sigmas_keep[i], \
                    slopes_keep[i] / sigmas_keep[i]
                strtxt = f'{ptxt[0]}\tgain/{timestep} days : {ptxt[1]:.2f} \t gain/year : {ptxt[2]:.2f} \t slope : {ptxt[3]:.2f} ' \
                        f'\t ' \
                        f'sigma :' \
                        f' {ptxt[4]:.2f} \t qual : {ptxt[5]:.2f}'
                printc(strtxt)
                printc(f'Sector : {sectors_keep[i]}')


        if doplot:
            plt.hist(growth_nothing, bins=30, histtype='step', color='r', label='Nothing', range=[0.6,1.8], fill=True,
                    alpha =0.5)
            plt.hist(growth_strategy, bins=30, histtype='step', color='g', label='Strategy', alpha = 0.5, fill=True, range=[0.6,1.8])
            plt.legend()
            plt.show()


        mean_growth_nothing = np.nanprod(growth_nothing)**(1/nyears)
        mean_growth_strategy = np.nanprod(growth_strategy)**(1/nyears)

        t_year = (t_decision - Time('2000-01-01').mjd)/365.25+2000
        if doplot:
            plt.plot(t_year,growth_nothing,'r')
            plt.plot(t_year,growth_strategy,'g')
            plt.show()

        fit_nothing = np.polyfit(t_year,np.log(np.cumproduct(growth_nothing)),1)
        rms_nothing = np.nanstd(np.log(np.cumproduct(growth_nothing)) - np.polyval(fit_nothing,t_year))

        fit_strategy = np.polyfit(t_year,np.log(np.cumproduct(growth_strategy)),1)
        rms_strategy = np.nanstd(np.log(np.cumproduct(growth_strategy)) - np.polyval(fit_strategy,t_year))

        printc(f'Mean yearly growth doing nothing : {mean_growth_nothing:.2f}, rms : {rms_nothing:.2f}')
        printc(f'Mean yearly growth with strategy : {mean_growth_strategy:.2f}, rms : {rms_strategy:.2f}')

        tbl['mean_growth_nothing'][nth_simu] = mean_growth_nothing
        tbl['mean_growth_strategy'][nth_simu] = mean_growth_strategy
        tbl['rms_nothing'][nth_simu] = rms_nothing
        tbl['rms_strategy'][nth_simu] = rms_strategy

        uyear = np.unique(t_year.astype(int))
        year_mean_nothing = np.zeros(len(uyear))
        year_mean_strategy = np.zeros(len(uyear))
        for i, year in enumerate(uyear):
            mask = t_year.astype(int) == year
            year_mean_nothing[i] = np.prod(growth_nothing[mask])
            year_mean_strategy[i] = np.prod(growth_strategy[mask])

        if doplot:
            plt.plot(t_year,np.log(np.cumproduct(growth_strategy)))
            plt.plot(t_year,np.log(np.cumproduct(growth_nothing)))
            plt.show()


            plt.step(uyear,year_mean_nothing,'r',label='Nothing')
            plt.step(uyear,year_mean_strategy,'g',label='Strategy')
            plt.legend()
            plt.show()

            year_mean_strategy[year_mean_strategy<0.5] = 0.5
            year_mean_nothing[year_mean_nothing<0.5] = 0.5
            year_mean_nothing[year_mean_nothing>2] = 2
            year_mean_strategy[year_mean_strategy>2] = 2
            plt.hist(year_mean_strategy, bins=10, histtype='step', color='g', label='Strategy', range=[0.4,2], fill=True,
                    alpha =0.5)
            plt.hist(year_mean_nothing, bins=10, histtype='step', color='r', label='Nothing', range=[0.4,2], fill=True,
                    alpha =0.5)
            plt.legend()
            plt.show()


        printc(f'Fraction {np.sum(growth_strategy>1)/len(growth_strategy):.2f} of {timestep} days with positive growth [{np.sum(growth_strategy>1)}/{len(growth_strategy)}] -- '
                f'strategy')
        printc(f'Fraction {np.sum(growth_nothing>1)/len(growth_nothing):.2f} of {timestep} days with positive growth [{np.sum(growth_nothing>1)}/{len(growth_nothing)}] -- nothing')

        printc(f'Fraction {np.sum(year_mean_strategy>1)/len(year_mean_strategy):.2f} of yearly with positive growth [{np.sum(year_mean_strategy>1)}/{len(year_mean_strategy)}] -- '
            f'strategy')
        printc(f'Fraction {np.sum(year_mean_nothing>1)/len(year_mean_nothing):.2f} of yearly with positive growth [{np.sum(year_mean_nothing>1)}/{len(year_mean_nothing)}] -- nothing')



        # We plot the growth in the last 2*lookback for the best stocks
        # We want the same color per sector
        usector = np.unique(sectors_keep)
        colors = plt.cm.coolwarm(np.linspace(0, 1, len(usector)))
        ord = np.argsort(sectors_keep)
        gains_keep = gains_keep[ord]
        slopes_keep = slopes_keep[ord]
        sigmas_keep = sigmas_keep[ord]
        keys_keep = keys_keep[ord]
        sectors_keep = sectors_keep[ord]

        if False:
            for i in range(len(keys_keep)):
                key = keys_keep[i]
                tbl1 = all_sp500[key]
                #
                keep = tbl1['mjd'] > t_decision[imc]-2*lookback
                for key_tbl in tbl1.keys():
                    tbl1[key_tbl] = tbl1[key_tbl][keep]
                moy = np.mean(np.log(tbl1['Close']))

                info = bt.get_info(key)
                name = info['shortName']

                if (i % 2) == 0:
                    linestyle = '-'
                    linewidth = 1
                else:
                    linestyle = '--'
                    linewidth = 2

                plt.plot_date(tbl1['date'],(np.log(tbl1['Close']) - moy),label=f'{name} [{key}] {sectors_keep[i]}',
                            linestyle=linestyle,
                            color = colors[np.where(usector == sectors_keep[i])[0][0]], linewidth=linewidth,marker=None)
            plt.legend()
            plt.grid()
            plt.show()



if __name__ == "__main__":
    simu_sp500(nkeeps=[10], timesteps=[45], lookback_scales=[7], 
    method='slope', injection=30000, nyears=20)