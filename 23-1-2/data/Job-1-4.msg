1

   Abaqus Learning Edition 2022                                  Date 01-1-2023   Time 20:53:26
   

                                                                                 
                                                                                 
 STEP    1     INCREMENT     1     STEP TIME    0.00    


                        S T E P       1     S T A T I C   A N A L Y S I S


                                                                                          

     AUTOMATIC TIME CONTROL WITH -
          A SUGGESTED INITIAL TIME INCREMENT OF                1.000E-04
          AND A TOTAL TIME PERIOD OF                            1.00    
          THE MINIMUM TIME INCREMENT ALLOWED IS                1.000E-12
          THE MAXIMUM TIME INCREMENT ALLOWED IS                0.100    

     LINEAR EQUATION SOLVER TYPE         DIRECT SPARSE

 CONVERGENCE TOLERANCE PARAMETERS FOR FORCE    
     CRITERION FOR RESIDUAL FORCE     FOR A NONLINEAR PROBLEM          5.000E-03
     CRITERION FOR DISP.    CORRECTION IN A NONLINEAR PROBLEM          1.000E-02
     INITIAL VALUE OF TIME AVERAGE FORCE                               1.000E-02
     AVERAGE FORCE     IS TIME AVERAGE FORCE    
     ALTERNATE CRIT. FOR RESIDUAL FORCE     FOR A NONLINEAR PROBLEM    2.000E-02
     CRITERION FOR ZERO FORCE     RELATIVE TO TIME AVRG. FORCE         1.000E-05
     CRITERION FOR RESIDUAL FORCE     WHEN THERE IS ZERO FLUX          1.000E-05
     CRITERION FOR DISP.    CORRECTION WHEN THERE IS ZERO FLUX         1.000E-03
     CRITERION FOR RESIDUAL FORCE     FOR A LINEAR INCREMENT           1.000E-08
     FIELD CONVERSION RATIO                                             1.00    
     CRITERION FOR ZERO FORCE     REL. TO TIME AVRG. MAX. FORCE        1.000E-05
     CRITERION FOR ZERO DISP.    RELATIVE TO CHARACTERISTIC LENGTH     1.000E-08

 CONVERGENCE TOLERANCE PARAMETERS FOR MOMENT   
     CRITERION FOR RESIDUAL MOMENT    FOR A NONLINEAR PROBLEM          5.000E-03
     CRITERION FOR ROTATION CORRECTION IN A NONLINEAR PROBLEM          1.000E-02
     INITIAL VALUE OF TIME AVERAGE MOMENT                              1.000E-02
     AVERAGE MOMENT    IS TIME AVERAGE MOMENT   
     ALTERNATE CRIT. FOR RESIDUAL MOMENT    FOR A NONLINEAR PROBLEM    2.000E-02
     CRITERION FOR ZERO MOMENT    RELATIVE TO TIME AVRG. MOMENT        1.000E-05
     CRITERION FOR RESIDUAL MOMENT    WHEN THERE IS ZERO FLUX          1.000E-05
     CRITERION FOR ROTATION CORRECTION WHEN THERE IS ZERO FLUX         1.000E-03
     CRITERION FOR RESIDUAL MOMENT    FOR A LINEAR INCREMENT           1.000E-08
     FIELD CONVERSION RATIO                                             1.00    
     CRITERION FOR ZERO MOMENT    REL. TO TIME AVRG. MAX. MOMENT       1.000E-05

     VOLUMETRIC STRAIN COMPATIBILITY TOLERANCE FOR HYBRID SOLIDS       1.000E-05
     AXIAL STRAIN COMPATIBILITY TOLERANCE FOR HYBRID BEAMS             1.000E-05
     TRANS. SHEAR STRAIN COMPATIBILITY TOLERANCE FOR HYBRID BEAMS      1.000E-05
     SOFT CONTACT CONSTRAINT COMPATIBILITY TOLERANCE FOR P>P0          5.000E-03
     SOFT CONTACT CONSTRAINT COMPATIBILITY TOLERANCE FOR P=0.0         0.100    
     CONTACT FORCE ERROR TOLERANCE FOR CONVERT SDI=YES                 1.00    
     DISPLACEMENT COMPATIBILITY TOLERANCE FOR DCOUP ELEMENTS           1.000E-05
     ROTATION COMPATIBILITY TOLERANCE FOR DCOUP ELEMENTS               1.000E-05

 EQUILIBRIUM WILL BE CHECKED FOR SEVERE DISCONTINUITY ITERATIONS

 TIME INCREMENTATION CONTROL PARAMETERS:
     FIRST EQUILIBRIUM ITERATION FOR CONSECUTIVE DIVERGENCE CHECK              4
     EQUILIBRIUM ITERATION AT WHICH LOG. CONVERGENCE RATE CHECK BEGINS         8
     EQUILIBRIUM ITERATION AFTER WHICH ALTERNATE RESIDUAL IS USED              9
     MAXIMUM EQUILIBRIUM ITERATIONS ALLOWED                                   16
     EQUILIBRIUM ITERATION COUNT FOR CUT-BACK IN NEXT INCREMENT               10
     MAXIMUM EQUILIB. ITERS IN TWO INCREMENTS FOR TIME INCREMENT INCREASE      4
     MAXIMUM ITERATIONS FOR SEVERE DISCONTINUITIES                            50
     MAXIMUM ATTEMPTS ALLOWED IN AN INCREMENT                                  5
     MAXIMUM DISCON. ITERS IN TWO INCREMENTS FOR TIME INCREMENT INCREASE      50
     MAXIMUM CONTACT AUGMENTATIONS FOR *SURFACE BEHAVIOR,AUGMENTED LAGRANGE   50
     CUT-BACK FACTOR AFTER DIVERGENCE                                 0.2500    
     CUT-BACK FACTOR FOR TOO SLOW CONVERGENCE                         0.5000    
     CUT-BACK FACTOR AFTER TOO MANY EQUILIBRIUM ITERATIONS            0.7500    
     CUT-BACK FACTOR AFTER TOO MANY SEVERE DISCONTINUITY ITERATIONS   0.2500    
     CUT-BACK FACTOR AFTER PROBLEMS IN ELEMENT ASSEMBLY               0.2500    
     INCREASE FACTOR AFTER TWO INCREMENTS THAT CONVERGE QUICKLY        1.500    
     MAX. TIME INCREMENT INCREASE FACTOR ALLOWED                       1.500    
     MAX. TIME INCREMENT INCREASE FACTOR ALLOWED (DYNAMICS)            1.250    
     MAX. TIME INCREMENT INCREASE FACTOR ALLOWED (DIFFUSION)           2.000    
     MINIMUM TIME INCREMENT RATIO FOR EXTRAPOLATION TO OCCUR          0.1000    
     MAX. RATIO OF TIME INCREMENT TO STABILITY LIMIT                   1.000    
     FRACTION OF STABILITY LIMIT FOR NEW TIME INCREMENT               0.9500    
     TIME INCREMENT INCREASE FACTOR BEFORE A TIME POINT                1.000    

 CONTACT CONTROLS APPLIED TO ALL CONTACT PAIRS:
     GLOBAL STABILIZATION CONTROL IS NOT USED
     RELATIVE PENETRATION TOLERANCE FOR AUGMENTED LAGRANGE CONTACT:
          FINITE-SLIDING SURFACE-TO-SURFACE                            5.000E-02
          OTHERWISE                                                    1.000E-03

          PRINT OF INCREMENT NUMBER, TIME, ETC., EVERY    1  INCREMENTS

     THE MAXIMUM NUMBER OF INCREMENTS IN THIS STEP IS                   10000

     LINEAR EXTRAPOLATION WILL BE USED

     CHARACTERISTIC ELEMENT LENGTH      3.00    

     DETAILS REGARDING ACTUAL SOLUTION WAVEFRONT REQUESTED

     DETAILED OUTPUT OF DIAGNOSTICS TO DATABASE REQUESTED

     PRINT OF INCREMENT NUMBER, TIME, ETC., TO THE MESSAGE FILE EVERY     1  INCREMENTS

     COLLECTING MODEL CONSTRAINT INFORMATION FOR OVERCONSTRAINT CHECKS

     COLLECTING STEP CONSTRAINT INFORMATION FOR OVERCONSTRAINT CHECKS


  INCREMENT     1 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04
 
	SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     297
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         3.304e+05

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      2.448E-05   TIME AVG. FORCE       2.448E-05
 LARGEST RESIDUAL FORCE            -9.501E-06   AT NODE        106   DOF  2
   INSTANCE: MERGE-1-1                                                                       
 LARGEST INCREMENT OF DISP.         2.616E-05   AT NODE         39   DOF  1
   INSTANCE: MERGE-1-1                                                                       
 LARGEST CORRECTION TO DISP.        2.616E-05   AT NODE         39   DOF  1
   INSTANCE: MERGE-1-1                                                                       
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      3.000E-02
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION      2.076E-38   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     2.076E-38   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     297
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         4.085e+05

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     2


 AVERAGE FORCE                      2.389E-05   TIME AVG. FORCE       2.389E-05
 LARGEST RESIDUAL FORCE            -1.846E-07   AT NODE        106   DOF  2
   INSTANCE: MERGE-1-1                                                                       
 LARGEST INCREMENT OF DISP.         2.615E-05   AT NODE         39   DOF  1
   INSTANCE: MERGE-1-1                                                                       
 LARGEST CORRECTION TO DISP.        1.050E-06   AT NODE         99   DOF  2
   INSTANCE: MERGE-1-1                                                                       
          FORCE     EQUILIBRIUM NOT ACHIEVED WITHIN TOLERANCE.

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      3.000E-02
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION      1.541E-38   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION    -5.998E-39   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
 
	SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     297
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         4.085e+05

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     3


 AVERAGE FORCE                      2.432E-05   TIME AVG. FORCE       2.432E-05
 LARGEST RESIDUAL FORCE            -7.512E-11   AT NODE        106   DOF  2
   INSTANCE: MERGE-1-1                                                                       
 LARGEST INCREMENT OF DISP.         2.615E-05   AT NODE         39   DOF  1
   INSTANCE: MERGE-1-1                                                                       
 LARGEST CORRECTION TO DISP.        2.121E-08   AT NODE         99   DOF  2
   INSTANCE: MERGE-1-1                                                                       
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      3.000E-02
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION      1.541E-38   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     5.770E-42   AT NODE          1   DOF  4
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 ITERATION SUMMARY FOR THE INCREMENT:   3 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  3 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  1.000E-04
 STEP TIME COMPLETED       1.000E-04,  TOTAL TIME COMPLETED        1.000E-04


  INCREMENT     2 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.000E-04
 
	SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     297
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         4.085e+05

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      3.715E-04   TIME AVG. FORCE       1.979E-04
 LARGEST RESIDUAL FORCE            -5.653E-08   AT NODE        106   DOF  2
   INSTANCE: MERGE-1-1                                                                       
 LARGEST INCREMENT OF DISP.         2.617E-05   AT NODE         39   DOF  1
   INSTANCE: MERGE-1-1                                                                       
 LARGEST CORRECTION TO DISP.       -1.837E-06   AT NODE         99   DOF  2
   INSTANCE: MERGE-1-1                                                                       
 ESTIMATE OF DISP.    CORRECTION    8.242E-09
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION      3.088E-38   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.547E-38   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
          TIME INCREMENT MAY NOW INCREASE TO    1.500E-04

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.000E-04,  FRACTION OF STEP COMPLETED  2.000E-04
 STEP TIME COMPLETED       2.000E-04,  TOTAL TIME COMPLETED        2.000E-04


  INCREMENT     3 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.500E-04
 
	SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     297
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         4.085e+05

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      6.585E-04   TIME AVG. FORCE       3.514E-04
 LARGEST RESIDUAL FORCE            -8.962E-10   AT NODE        106   DOF  2
   INSTANCE: MERGE-1-1                                                                       
 LARGEST INCREMENT OF DISP.         3.925E-05   AT NODE         39   DOF  1
   INSTANCE: MERGE-1-1                                                                       
 LARGEST CORRECTION TO DISP.        8.958E-07   AT NODE         99   DOF  2
   INSTANCE: MERGE-1-1                                                                       
 ESTIMATE OF DISP.    CORRECTION    1.362E-10
 FORCE     EQUILIB. ACCEPTED BASED ON SMALL RESIDUAL AND ESTIMATED CORRECTION

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION      5.404E-38   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     7.715E-39   AT NODE          2   DOF  6
   INSTANCE: MERGE-1-1                                                                       
          THERE IS ZERO MOMENT    EVERYWHERE
          TIME INCREMENT MAY NOW INCREASE TO    2.250E-04

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.500E-04,  FRACTION OF STEP COMPLETED  3.500E-04
 STEP TIME COMPLETED       3.500E-04,  TOTAL TIME COMPLETED        3.500E-04


  INCREMENT     4 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  2.250E-04
 
	SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     297
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         4.085e+05

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.096E-03   TIME AVG. FORCE       5.376E-04
 LARGEST RESIDUAL FORCE            -8.415E-12   AT NODE        106   DOF  2
   INSTANCE: MERGE-1-1                                                                       
 LARGEST INCREMENT OF DISP.         5.887E-05   AT NODE         39   DOF  1
   INSTANCE: MERGE-1-1                                                                       
 LARGEST CORRECTION TO DISP.        3.365E-07   AT NODE         99   DOF  2
   INSTANCE: MERGE-1-1                                                                       
          THE FORCE     EQUILIBRIUM EQUATIONS HAVE CONVERGED

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION      8.878E-38   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     7.726E-39   AT NODE          2   DOF  6
   INSTANCE: MERGE-1-1                                                                       
          THERE IS ZERO MOMENT    EVERYWHERE
          TIME INCREMENT MAY NOW INCREASE TO    3.375E-04

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  2.250E-04,  FRACTION OF STEP COMPLETED  5.750E-04
 STEP TIME COMPLETED       5.750E-04,  TOTAL TIME COMPLETED        5.750E-04


  INCREMENT     5 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  3.375E-04
 
	SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     297
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         4.085e+05

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.746E-03   TIME AVG. FORCE       7.792E-04
 LARGEST RESIDUAL FORCE            -1.652E-14   AT NODE        106   DOF  2
   INSTANCE: MERGE-1-1                                                                       
 LARGEST INCREMENT OF DISP.         8.830E-05   AT NODE         39   DOF  1
   INSTANCE: MERGE-1-1                                                                       
 LARGEST CORRECTION TO DISP.        5.802E-08   AT NODE         99   DOF  2
   INSTANCE: MERGE-1-1                                                                       
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION      1.409E-37   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     7.722E-39   AT NODE          2   DOF  6
   INSTANCE: MERGE-1-1                                                                       
          THERE IS ZERO MOMENT    EVERYWHERE
          TIME INCREMENT MAY NOW INCREASE TO    5.062E-04

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  3.375E-04,  FRACTION OF STEP COMPLETED  9.125E-04
 STEP TIME COMPLETED       9.125E-04,  TOTAL TIME COMPLETED        9.125E-04


  INCREMENT     6 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  5.062E-04
 
	SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     297
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         4.085e+05

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      2.714E-03   TIME AVG. FORCE       1.102E-03
 LARGEST RESIDUAL FORCE             3.166E-17   AT NODE         53   DOF  2
   INSTANCE: MERGE-1-1                                                                       
 LARGEST INCREMENT OF DISP.         1.324E-04   AT NODE         39   DOF  1
   INSTANCE: MERGE-1-1                                                                       
 LARGEST CORRECTION TO DISP.        9.448E-09   AT NODE         99   DOF  2
   INSTANCE: MERGE-1-1                                                                       
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION      2.191E-37   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     7.720E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
          TIME INCREMENT MAY NOW INCREASE TO    7.594E-04

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  5.062E-04,  FRACTION OF STEP COMPLETED  1.419E-03
 STEP TIME COMPLETED       1.419E-03,  TOTAL TIME COMPLETED        1.419E-03


  INCREMENT     7 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  7.594E-04
 
	SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     297
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         4.085e+05

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      4.167E-03   TIME AVG. FORCE       1.540E-03
 LARGEST RESIDUAL FORCE             6.202E-17   AT NODE         13   DOF  2
   INSTANCE: MERGE-1-1                                                                       
 LARGEST INCREMENT OF DISP.         1.987E-04   AT NODE         39   DOF  1
   INSTANCE: MERGE-1-1                                                                       
 LARGEST CORRECTION TO DISP.        1.505E-09   AT NODE         99   DOF  2
   INSTANCE: MERGE-1-1                                                                       
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION      3.363E-37   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     7.720E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
          TIME INCREMENT MAY NOW INCREASE TO    1.139E-03

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  7.594E-04,  FRACTION OF STEP COMPLETED  2.178E-03
 STEP TIME COMPLETED       2.178E-03,  TOTAL TIME COMPLETED        2.178E-03


  INCREMENT     8 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.139E-03
 
	SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     297
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         4.085e+05

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      6.346E-03   TIME AVG. FORCE       2.140E-03
 LARGEST RESIDUAL FORCE             1.050E-16   AT NODE         37   DOF  1
   INSTANCE: MERGE-1-1                                                                       
 LARGEST INCREMENT OF DISP.         2.980E-04   AT NODE         39   DOF  1
   INSTANCE: MERGE-1-1                                                                       
 LARGEST CORRECTION TO DISP.        2.347E-10   AT NODE         99   DOF  2
   INSTANCE: MERGE-1-1                                                                       
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION      5.122E-37   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     7.720E-39   AT NODE          2   DOF  6
   INSTANCE: MERGE-1-1                                                                       
          THERE IS ZERO MOMENT    EVERYWHERE
          TIME INCREMENT MAY NOW INCREASE TO    1.709E-03

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.139E-03,  FRACTION OF STEP COMPLETED  3.317E-03
 STEP TIME COMPLETED       3.317E-03,  TOTAL TIME COMPLETED        3.317E-03


  INCREMENT     9 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.709E-03
 
	SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     297
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         4.085e+05

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      9.615E-03   TIME AVG. FORCE       2.971E-03
 LARGEST RESIDUAL FORCE            -1.095E-16   AT NODE         65   DOF  2
   INSTANCE: MERGE-1-1                                                                       
 LARGEST INCREMENT OF DISP.         4.470E-04   AT NODE         39   DOF  1
   INSTANCE: MERGE-1-1                                                                       
 LARGEST CORRECTION TO DISP.        3.609E-11   AT NODE         99   DOF  2
   INSTANCE: MERGE-1-1                                                                       
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION      7.760E-37   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     7.720E-39   AT NODE          2   DOF  6
   INSTANCE: MERGE-1-1                                                                       
          THERE IS ZERO MOMENT    EVERYWHERE
          TIME INCREMENT MAY NOW INCREASE TO    2.563E-03

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.709E-03,  FRACTION OF STEP COMPLETED  5.026E-03
 STEP TIME COMPLETED       5.026E-03,  TOTAL TIME COMPLETED        5.026E-03


  INCREMENT    10 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  2.563E-03
 
	SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     297
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         4.085e+05

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      1.452E-02   TIME AVG. FORCE       4.126E-03
 LARGEST RESIDUAL FORCE             3.770E-16   AT NODE          9   DOF  2
   INSTANCE: MERGE-1-1                                                                       
 LARGEST INCREMENT OF DISP.         6.705E-04   AT NODE         39   DOF  1
   INSTANCE: MERGE-1-1                                                                       
 LARGEST CORRECTION TO DISP.        5.468E-12   AT NODE         99   DOF  2
   INSTANCE: MERGE-1-1                                                                       
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.000E-02
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION      1.172E-36   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     7.720E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
          TIME INCREMENT MAY NOW INCREASE TO    3.844E-03

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  2.563E-03,  FRACTION OF STEP COMPLETED  7.589E-03
 STEP TIME COMPLETED       7.589E-03,  TOTAL TIME COMPLETED        7.589E-03


  INCREMENT    11 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  3.844E-03
 
	SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     297
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         4.085e+05

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      2.187E-02   TIME AVG. FORCE       5.739E-03
 LARGEST RESIDUAL FORCE            -1.761E-16   AT NODE         65   DOF  2
   INSTANCE: MERGE-1-1                                                                       
 LARGEST INCREMENT OF DISP.         1.006E-03   AT NODE         39   DOF  1
   INSTANCE: MERGE-1-1                                                                       
 LARGEST CORRECTION TO DISP.        9.013E-13   AT NODE         99   DOF  2
   INSTANCE: MERGE-1-1                                                                       
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.238E-02
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION      1.765E-36   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     7.720E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
          TIME INCREMENT MAY NOW INCREASE TO    5.767E-03

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  3.844E-03,  FRACTION OF STEP COMPLETED  1.143E-02
 STEP TIME COMPLETED       1.143E-02,  TOTAL TIME COMPLETED        1.143E-02


  INCREMENT    12 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  5.767E-03
 
	SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     297
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         4.085e+05

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      3.290E-02   TIME AVG. FORCE       8.003E-03
 LARGEST RESIDUAL FORCE            -3.712E-16   AT NODE         87   DOF  2
   INSTANCE: MERGE-1-1                                                                       
 LARGEST INCREMENT OF DISP.         1.509E-03   AT NODE         39   DOF  1
   INSTANCE: MERGE-1-1                                                                       
 LARGEST CORRECTION TO DISP.        9.887E-14   AT NODE         99   DOF  2
   INSTANCE: MERGE-1-1                                                                       
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      1.722E-02
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION      2.656E-36   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     7.720E-39   AT NODE          2   DOF  6
   INSTANCE: MERGE-1-1                                                                       
          THERE IS ZERO MOMENT    EVERYWHERE
          TIME INCREMENT MAY NOW INCREASE TO    8.650E-03

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  5.767E-03,  FRACTION OF STEP COMPLETED  1.720E-02
 STEP TIME COMPLETED       1.720E-02,  TOTAL TIME COMPLETED        1.720E-02


  INCREMENT    13 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  8.650E-03
 
	SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     297
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         4.085e+05

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      4.945E-02   TIME AVG. FORCE       1.119E-02
 LARGEST RESIDUAL FORCE             6.991E-16   AT NODE         28   DOF  1
   INSTANCE: MERGE-1-1                                                                       
 LARGEST INCREMENT OF DISP.         2.263E-03   AT NODE         39   DOF  1
   INSTANCE: MERGE-1-1                                                                       
 LARGEST CORRECTION TO DISP.        9.949E-15   AT NODE         99   DOF  2
   INSTANCE: MERGE-1-1                                                                       
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      2.401E-02
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION      3.991E-36   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     7.720E-39   AT NODE          2   DOF  6
   INSTANCE: MERGE-1-1                                                                       
          THERE IS ZERO MOMENT    EVERYWHERE
          TIME INCREMENT MAY NOW INCREASE TO    1.297E-02

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  8.650E-03,  FRACTION OF STEP COMPLETED  2.585E-02
 STEP TIME COMPLETED       2.585E-02,  TOTAL TIME COMPLETED        2.585E-02


  INCREMENT    14 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.297E-02
 
	SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     297
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         4.085e+05

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      7.427E-02   TIME AVG. FORCE       1.570E-02
 LARGEST RESIDUAL FORCE             7.425E-16   AT NODE          4   DOF  2
   INSTANCE: MERGE-1-1                                                                       
 LARGEST INCREMENT OF DISP.         3.395E-03   AT NODE         39   DOF  1
   INSTANCE: MERGE-1-1                                                                       
 LARGEST CORRECTION TO DISP.        9.892E-16   AT NODE         99   DOF  2
   INSTANCE: MERGE-1-1                                                                       
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      3.357E-02
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION      5.994E-36   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     7.720E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
          TIME INCREMENT MAY NOW INCREASE TO    1.946E-02

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.297E-02,  FRACTION OF STEP COMPLETED  3.882E-02
 STEP TIME COMPLETED       3.882E-02,  TOTAL TIME COMPLETED        3.882E-02


  INCREMENT    15 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  1.946E-02
 
	SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     297
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         4.085e+05

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.112       TIME AVG. FORCE       2.208E-02
 LARGEST RESIDUAL FORCE             1.400E-15   AT NODE         31   DOF  1
   INSTANCE: MERGE-1-1                                                                       
 LARGEST INCREMENT OF DISP.         5.092E-03   AT NODE         39   DOF  1
   INSTANCE: MERGE-1-1                                                                       
 LARGEST CORRECTION TO DISP.        9.863E-17   AT NODE         99   DOF  2
   INSTANCE: MERGE-1-1                                                                       
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      4.709E-02
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION      8.999E-36   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     7.720E-39   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE
          TIME INCREMENT MAY NOW INCREASE TO    2.919E-02

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  1.946E-02,  FRACTION OF STEP COMPLETED  5.829E-02
 STEP TIME COMPLETED       5.829E-02,  TOTAL TIME COMPLETED        5.829E-02


  INCREMENT    16 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  2.919E-02
 
	SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     297
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         4.085e+05

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.167       TIME AVG. FORCE       3.116E-02
 LARGEST RESIDUAL FORCE            -3.247E-15   AT NODE         29   DOF  1
   INSTANCE: MERGE-1-1                                                                       
 LARGEST INCREMENT OF DISP.         7.638E-03   AT NODE         39   DOF  1
   INSTANCE: MERGE-1-1                                                                       
 LARGEST CORRECTION TO DISP.        9.512E-18   AT NODE         99   DOF  2
   INSTANCE: MERGE-1-1                                                                       
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      6.625E-02
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION      1.351E-35   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     7.720E-39   AT NODE          2   DOF  6
   INSTANCE: MERGE-1-1                                                                       
          THERE IS ZERO MOMENT    EVERYWHERE
          TIME INCREMENT MAY NOW INCREASE TO    4.379E-02

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  2.919E-02,  FRACTION OF STEP COMPLETED  8.748E-02
 STEP TIME COMPLETED       8.748E-02,  TOTAL TIME COMPLETED        8.748E-02


  INCREMENT    17 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  4.379E-02
 
	SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     297
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         4.085e+05

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.251       TIME AVG. FORCE       4.410E-02
 LARGEST RESIDUAL FORCE             5.010E-15   AT NODE         29   DOF  1
   INSTANCE: MERGE-1-1                                                                       
 LARGEST INCREMENT OF DISP.         1.146E-02   AT NODE         39   DOF  1
   INSTANCE: MERGE-1-1                                                                       
 LARGEST CORRECTION TO DISP.       -6.074E-18   AT NODE        115   DOF  3
   INSTANCE: MERGE-1-1                                                                       
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      9.349E-02
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION      2.027E-35   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     7.720E-39   AT NODE          2   DOF  6
   INSTANCE: MERGE-1-1                                                                       
          THERE IS ZERO MOMENT    EVERYWHERE
          TIME INCREMENT MAY NOW INCREASE TO    6.568E-02

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  4.379E-02,  FRACTION OF STEP COMPLETED  0.131    
 STEP TIME COMPLETED       0.131    ,  TOTAL TIME COMPLETED        0.131    


  INCREMENT    18 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  6.568E-02
 
	SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     297
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         4.085e+05

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.377       TIME AVG. FORCE       6.259E-02
 LARGEST RESIDUAL FORCE             4.718E-15   AT NODE         35   DOF  1
   INSTANCE: MERGE-1-1                                                                       
 LARGEST INCREMENT OF DISP.         1.718E-02   AT NODE         39   DOF  1
   INSTANCE: MERGE-1-1                                                                       
 LARGEST CORRECTION TO DISP.        2.015E-17   AT NODE         27   DOF  1
   INSTANCE: MERGE-1-1                                                                       
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      0.132    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION      3.041E-35   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     7.720E-39   AT NODE          2   DOF  6
   INSTANCE: MERGE-1-1                                                                       
          THERE IS ZERO MOMENT    EVERYWHERE
          TIME INCREMENT MAY NOW INCREASE TO    9.853E-02

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  6.568E-02,  FRACTION OF STEP COMPLETED  0.197    
 STEP TIME COMPLETED       0.197    ,  TOTAL TIME COMPLETED        0.197    


  INCREMENT    19 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  9.853E-02
 
	SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     297
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         4.085e+05

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.565       TIME AVG. FORCE       8.904E-02
 LARGEST RESIDUAL FORCE             8.986E-15   AT NODE         31   DOF  1
   INSTANCE: MERGE-1-1                                                                       
 LARGEST INCREMENT OF DISP.         2.578E-02   AT NODE         39   DOF  1
   INSTANCE: MERGE-1-1                                                                       
 LARGEST CORRECTION TO DISP.       -3.709E-17   AT NODE        109   DOF  1
   INSTANCE: MERGE-1-1                                                                       
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      0.188    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION      4.562E-35   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     7.720E-39   AT NODE          2   DOF  6
   INSTANCE: MERGE-1-1                                                                       
          THERE IS ZERO MOMENT    EVERYWHERE
          TIME INCREMENT MAY NOW INCREASE TO    0.100    

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  9.853E-02,  FRACTION OF STEP COMPLETED  0.295    
 STEP TIME COMPLETED       0.295    ,  TOTAL TIME COMPLETED        0.295    


  INCREMENT    20 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.100    
 
	SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     297
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         4.085e+05

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =  0.10000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =  0.10000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.757       TIME AVG. FORCE       0.122    
 LARGEST RESIDUAL FORCE             8.216E-15   AT NODE         41   DOF  1
   INSTANCE: MERGE-1-1                                                                       
 LARGEST INCREMENT OF DISP.         2.616E-02   AT NODE         39   DOF  1
   INSTANCE: MERGE-1-1                                                                       
 LARGEST CORRECTION TO DISP.       -7.264E-17   AT NODE         30   DOF  1
   INSTANCE: MERGE-1-1                                                                       
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      0.267    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION      6.106E-35   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.476E-35   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.100    ,  FRACTION OF STEP COMPLETED  0.395    
 STEP TIME COMPLETED       0.395    ,  TOTAL TIME COMPLETED        0.395    


  INCREMENT    21 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.100    
 
	SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     297
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         4.085e+05

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                      0.948       TIME AVG. FORCE       0.162    
 LARGEST RESIDUAL FORCE            -7.105E-15   AT NODE         56   DOF  2
   INSTANCE: MERGE-1-1                                                                       
 LARGEST INCREMENT OF DISP.         2.616E-02   AT NODE         39   DOF  1
   INSTANCE: MERGE-1-1                                                                       
 LARGEST CORRECTION TO DISP.       -5.187E-17   AT NODE         83   DOF  2
   INSTANCE: MERGE-1-1                                                                       
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      0.367    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION      7.650E-35   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.544E-35   AT NODE          2   DOF  6
   INSTANCE: MERGE-1-1                                                                       
          THERE IS ZERO MOMENT    EVERYWHERE

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.100    ,  FRACTION OF STEP COMPLETED  0.495    
 STEP TIME COMPLETED       0.495    ,  TOTAL TIME COMPLETED        0.495    


  INCREMENT    22 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.100    
 
	SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     297
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         4.085e+05

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.14       TIME AVG. FORCE       0.206    
 LARGEST RESIDUAL FORCE             7.619E-15   AT NODE         70   DOF  2
   INSTANCE: MERGE-1-1                                                                       
 LARGEST INCREMENT OF DISP.         2.616E-02   AT NODE         39   DOF  1
   INSTANCE: MERGE-1-1                                                                       
 LARGEST CORRECTION TO DISP.       -5.814E-17   AT NODE        109   DOF  2
   INSTANCE: MERGE-1-1                                                                       
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      0.485    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION      9.194E-35   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.544E-35   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.100    ,  FRACTION OF STEP COMPLETED  0.595    
 STEP TIME COMPLETED       0.595    ,  TOTAL TIME COMPLETED        0.595    


  INCREMENT    23 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.100    
 
	SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     297
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         4.085e+05

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.33       TIME AVG. FORCE       0.255    
 LARGEST RESIDUAL FORCE            -7.327E-15   AT NODE          7   DOF  2
   INSTANCE: MERGE-1-1                                                                       
 LARGEST INCREMENT OF DISP.         2.616E-02   AT NODE         39   DOF  1
   INSTANCE: MERGE-1-1                                                                       
 LARGEST CORRECTION TO DISP.       -4.473E-17   AT NODE         33   DOF  1
   INSTANCE: MERGE-1-1                                                                       
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      0.618    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION      1.074E-34   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.544E-35   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.100    ,  FRACTION OF STEP COMPLETED  0.695    
 STEP TIME COMPLETED       0.695    ,  TOTAL TIME COMPLETED        0.695    


  INCREMENT    24 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.100    
 
	SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     297
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         4.085e+05

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.52       TIME AVG. FORCE       0.308    
 LARGEST RESIDUAL FORCE             1.227E-14   AT NODE         69   DOF  2
   INSTANCE: MERGE-1-1                                                                       
 LARGEST INCREMENT OF DISP.         2.616E-02   AT NODE         39   DOF  1
   INSTANCE: MERGE-1-1                                                                       
 LARGEST CORRECTION TO DISP.       -9.723E-17   AT NODE        107   DOF  3
   INSTANCE: MERGE-1-1                                                                       
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      0.765    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION      1.228E-34   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.544E-35   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.100    ,  FRACTION OF STEP COMPLETED  0.795    
 STEP TIME COMPLETED       0.795    ,  TOTAL TIME COMPLETED        0.795    


  INCREMENT    25 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.100    
 
	SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     297
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         4.085e+05

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.71       TIME AVG. FORCE       0.364    
 LARGEST RESIDUAL FORCE             7.327E-15   AT NODE         53   DOF  2
   INSTANCE: MERGE-1-1                                                                       
 LARGEST INCREMENT OF DISP.         2.616E-02   AT NODE         39   DOF  1
   INSTANCE: MERGE-1-1                                                                       
 LARGEST CORRECTION TO DISP.        1.228E-16   AT NODE        107   DOF  3
   INSTANCE: MERGE-1-1                                                                       
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT      0.923    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION      1.383E-34   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.544E-35   AT NODE          2   DOF  6
   INSTANCE: MERGE-1-1                                                                       
          THERE IS ZERO MOMENT    EVERYWHERE

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.100    ,  FRACTION OF STEP COMPLETED  0.895    
 STEP TIME COMPLETED       0.895    ,  TOTAL TIME COMPLETED        0.895    


  INCREMENT    26 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  0.100    
 
	SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     297
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         4.085e+05

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.90       TIME AVG. FORCE       0.423    
 LARGEST RESIDUAL FORCE             1.521E-14   AT NODE         31   DOF  1
   INSTANCE: MERGE-1-1                                                                       
 LARGEST INCREMENT OF DISP.         2.616E-02   AT NODE         39   DOF  1
   INSTANCE: MERGE-1-1                                                                       
 LARGEST CORRECTION TO DISP.        1.796E-16   AT NODE         83   DOF  2
   INSTANCE: MERGE-1-1                                                                       
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       1.09    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION      1.537E-34   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.544E-35   AT NODE          2   DOF  6
   INSTANCE: MERGE-1-1                                                                       
          THERE IS ZERO MOMENT    EVERYWHERE

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  0.100    ,  FRACTION OF STEP COMPLETED  0.995    
 STEP TIME COMPLETED       0.995    ,  TOTAL TIME COMPLETED        0.995    


  INCREMENT    27 STARTS. ATTEMPT NUMBER  1, TIME INCREMENT  4.522E-03
 
	SYMMETRIC DIRECT SPARSE SOLVER RUNNING ON
 	1 HOST:        1 MPI RANK  x 1 THREAD
 	NUMBER OF EQUATIONS:     297
 	NUMBER OF RHS:           1
 	NUMBER OF FLOPS:         4.085e+05

     CHECK POINT  START OF SOLVER     

     CHECK POINT  END OF SOLVER       

       ELAPSED USER TIME (SEC)      =   0.0000    
       ELAPSED SYSTEM TIME (SEC)    =   0.0000    
       ELAPSED TOTAL CPU TIME (SEC) =   0.0000    
       ELAPSED WALLCLOCK TIME (SEC) =            0

               CONVERGENCE CHECKS FOR EQUILIBRIUM ITERATION     1


 AVERAGE FORCE                       1.91       TIME AVG. FORCE       0.478    
 LARGEST RESIDUAL FORCE             5.551E-15   AT NODE         53   DOF  2
   INSTANCE: MERGE-1-1                                                                       
 LARGEST INCREMENT OF DISP.         1.183E-03   AT NODE         39   DOF  1
   INSTANCE: MERGE-1-1                                                                       
 LARGEST CORRECTION TO DISP.        6.491E-17   AT NODE         85   DOF  2
   INSTANCE: MERGE-1-1                                                                       
          THE FORCE     EQUILIBRIUM RESPONSE WAS LINEAR IN THIS INCREMENT

 AVERAGE MOMENT                      0.00       TIME AVG. MOMENT       1.27    
                                                (FROM FORCE     FIELD)
 ALL MOMENT    RESIDUALS ARE ZERO
 LARGEST INCREMENT OF ROTATION      1.544E-34   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
 LARGEST CORRECTION TO ROTATION     1.475E-34   AT NODE          1   DOF  6
   INSTANCE: {assembly node}                                                                 
          THERE IS ZERO MOMENT    EVERYWHERE

 ITERATION SUMMARY FOR THE INCREMENT:   1 TOTAL ITERATIONS, OF WHICH
   0 ARE SEVERE DISCONTINUITY ITERATIONS AND  1 ARE EQUILIBRIUM ITERATIONS.

 TIME INCREMENT COMPLETED  4.522E-03,  FRACTION OF STEP COMPLETED   1.00    
 STEP TIME COMPLETED        1.00    ,  TOTAL TIME COMPLETED         1.00    


          THE ANALYSIS HAS BEEN COMPLETED



     ANALYSIS SUMMARY:
     TOTAL OF         27  INCREMENTS
                       0  CUTBACKS IN AUTOMATIC INCREMENTATION
                      29  ITERATIONS INCLUDING CONTACT ITERATIONS IF PRESENT
                      29  PASSES THROUGH THE EQUATION SOLVER OF WHICH 
                      29  INVOLVE MATRIX DECOMPOSITION, INCLUDING
                       0  DECOMPOSITION(S) OF THE MASS MATRIX
                       1  REORDERING OF EQUATIONS TO MINIMIZE WAVEFRONT
                       0  ADDITIONAL RESIDUAL EVALUATIONS FOR LINE SEARCHES
                       0  ADDITIONAL OPERATOR EVALUATIONS FOR LINE SEARCHES
                       7  WARNING MESSAGES DURING USER INPUT PROCESSING
                       0  WARNING MESSAGES DURING ANALYSIS
                       0  ANALYSIS WARNINGS ARE NUMERICAL PROBLEM MESSAGES
                       0  ANALYSIS WARNINGS ARE NEGATIVE EIGENVALUE MESSAGES
                       0  ERROR MESSAGES



     JOB TIME SUMMARY
       USER TIME (SEC)      =      3.4    
       SYSTEM TIME (SEC)    =     0.20    
       TOTAL CPU TIME (SEC) =      3.6    
       WALLCLOCK TIME (SEC) =            4
       MEMORY PEAK (GB)     =            0
