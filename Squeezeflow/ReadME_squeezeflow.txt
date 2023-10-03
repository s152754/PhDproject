
###########################
######### general #########
###########################

The input parameters are defined in 'arguments'.
The addition of surface tension can be put on/off by adjusting the input parameter 'alpha' (0 < alpha < 1).
If 'alpha'=0, we have no effect of surface tension (straight fluid front).
If 'alpha'=1, we have maximum contribution of Laplace pressure (curved fluid front).

Run the code:
If you solely want to run the Newtonian model, make sure that 'cli.run(main)' is UNcommented
If you want to use this code in a sampler, uncomment 'cli.run(main)' and possibly the figures as well because it saves time.


###########################
##### Newtonian_sf.py #####
###########################

- The Newtonian squeeze flow assumes constant viscosity through time and throughout the fluid sample. 

- We use forward Euler to solve the height of the fluid layer as a function time. 
- 'hdot' is the vertical velocity of the upper layer.
- We use an adapative timestepper since initially we need more timesteps because of the fast deformation.


###########################
# TruncatedPowerLaw_sf.py #
###########################

- In the truncated power law squeeze flow, the viscosity throughout the sample varies as well as through time.
- We use an adaptive timestepper based on the analtical solution and factor 's'.
- The pressure gradient is solved using the Picard solver. 
- To save time, uncomment the postprocessing steps: 'pp.plot(t, r, dpdr, h, w1, w2)' and 'pp.sampledata(t, r, h)'
- The plots 'gamma.png' and 'viscosity.png' do not work, so make sure to uncomment them

