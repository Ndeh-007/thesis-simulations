### directions
1. modules are isolated
2. they all use the fluids and ps files

3. run each of their main functions. start with the python and then the julia. 


### cases
1. case 0.0: Base Case, => f0, p0
2. case 0.1: double the pump rate => f0
3. case 1.0: decrease yield stress of lead slurry
<!-- 4. case 1.1: increase density of tail slurry - done inside 1.0 => changes are made in fluid 2 -->
5. case 2.0: change eccentricity - standoff = 50%
6. case 2.1: change eccentricity - standoff = 25%
<!-- 7. case 2.2: change eccentricity - standoff = oscillate between 25% and 80% (use a sine wave) -->
8. case 3.0: density unstable - invert the fluid pump order of the lead and tail slurry
9. case 4.0: gap thickness - increase the gap thickness only by 75%