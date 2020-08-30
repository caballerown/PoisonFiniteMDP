function [z_final,time_final, R, P, B,U] = PoisonFiniteMDP_GradientFree_autotune(T,tbar,sbar,abar, cardS,alpha,mu1,mu2,mu3,Rhat,Bhat, Phat,epsilon, stepsizeR_init, stepsizeP_init, stepsizeB_init, n, zchangelimit, beta_p)

terminate_criteria = 0;
timestart = tic; %start timing
z_cand = 100000;

while terminate_criteria ==0
    stepsizelistR = [stepsizeR_init, beta_p*stepsizeR_init, stepsizeR_init, stepsizeR_init];
    stepsizelistP = [stepsizeP_init, stepsizeP_init, beta_p*stepsizeP_init, stepsizeP_init];
    stepsizelistB = [stepsizeB_init, stepsizeB_init, stepsizeB_init, beta_p*stepsizeB_init];
    
for direction = 1:4
    stepsizeR= stepsizelistR(direction);
    stepsizeP = stepsizelistP(direction);
    stepsizeB = stepsizelistB(direction);
    
    [z_check(direction),dummy,Rcheck{direction}, Pcheck{direction},Bcheck{direction}] = PoisonFiniteMDP_GradientFree(T,tbar,sbar,abar, cardS,alpha,mu1,mu2,mu3,Rhat,Bhat, Phat,epsilon, stepsizeR, stepsizeP, stepsizeB, n, zchangelimit);

end  %Loop to go thru all coord directions of step sizes

if z_cand <= min(z_check)
    terminate_criteria = 1; 
else
    [z_cand, z_cand_index] = min(z_check); 
    stepsizeR_init = stepsizelistR(z_cand_index);
    stepsizeP_init = stepsizelistP(z_cand_index);
    stepsizeB_init = stepsizelistB(z_cand_index);
    R = Pcheck{z_cand_index};
    P = Pcheck{z_cand_index};
    B = Pcheck{z_cand_index};
end

end %Loop to determine if no improving solution found

z_final = z_cand;
time_final = toc(timestart); %End timing;


end