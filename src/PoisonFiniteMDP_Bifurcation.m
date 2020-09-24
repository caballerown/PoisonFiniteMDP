function [z_final,time, R_iterminus1, P_iterminus1, B_iterminus1, U] = PoisonFiniteMDP_Bifurcation(T,tbar,sbar,abar, cardS,alpha,mu1,mu2,mu3,Rhat,Bhat, Phat,epsilon, stepsizeP, stepsizeR, zchangelimit)

%Description of Variables
%T = integer (# of time periods in finite-horizon MDP)
%tbar = integer (Between 1 & T - time for which action is being affected)
%sbar = integer (Between 1 & # states -- state action is being targeted)
%abar = integer (# of action at sbar, tbar desired to be taken)
%cardS = integer (# of states)
%alpha = term used in softmax and smoothmax approximations
%mu1 -- mu3: obj func weights for the respective changes to the parameters
%Rhat: baseline MDP immediate rewards
    %Rhat{s} = [rhat(s,action1),...,rhat(s,action_N)] 
    %        = [reward for action 1 @ s, ..., reward for action_N at s]
%Phat: baseline MDP transitions probs
    %Phat{s,a} = [phat(state 1|s,a), ...., phat(state card(S)-1|s,a)]
    %           = [transitioZn prob to state given s &a, ...] 
    %    *Note: Last state excluded to ensure sum of probabilities equal 1 later
%Bhat: baseline MDP terminal rewards
    %Bhat{s}= [b(1),...,b(card(S))] = [terminal reward state 1, ...]    
%epsilon: small positive # used in onj func to keep always defined
%stepsizeR: amount in random coord direction in R
%stepsizeP: "                                 " P
%zchangelimit: delta in paper, small negative # near zero -- stops algrithm

tic  %start timingg

%Initiate Decision variables to original MDP values
R=Rhat;
P=Phat;
B=Bhat;

%Find number of actions available at each state
for s = 1:cardS
    A{s} = size(Rhat{s},2);
end

%Initialize reward-to-go values to zero
U = zeros(T,cardS);

%Number of last state in MDP
k=cardS;

%Initialize Algorithm Specific Parameters
iter=1; %iteration counter
z(iter)=0;
iter=iter+1;
zchange = 10000;  %Set change in obj func b/w iter high at first


%%%%%%%
% Find first z-values from baseline then start while loop
%[z(iter),U,A]= LimitedBI(R,P,B,T,tbar,sbar,abar, cardS,alpha,mu1,mu2,mu3,Rhat,Bhat, Phat,epsilon);
[z(iter),U,A]= LimitedBISmooth(R,P,B,T,tbar,sbar,abar, cardS,alpha,mu1,mu2,mu3,Rhat,Bhat, Phat,epsilon);


while zchange>zchangelimit
    iter = iter+1;
    nu = mean(U(tbar+1,:)); %Find mean to separate states into two sets
    
    %Change the transition probs for each bifurcated set
    if isnan(z(iter-1)) ~= 1
    for j=1:k-1 
        if U(tbar+1,j) > nu %For "high-value" states
            if sum(P{sbar,abar}) + stepsizeP <=1 && P{sbar,abar}(j) + stepsizeP <=1
            %if P{sbar,abar}(j) + stepsizeP <=1
                P{sbar,abar}(j) = P{sbar,abar}(j) + stepsizeP;
                for a = 1: A{sbar}
                    if a~= abar && sum(P{sbar,a}) - stepsizeP >= 0 && P{sbar,a}(j) - stepsizeP >= 0
                    %if a~= abar && P{sbar,a}(j) - stepsizeP >= 0
                        P{sbar,a}(j) = P{sbar,a}(j) - stepsizeP;
                    end
                end
            end
        else %For "low-value" states do opposite of above
            if sum(P{sbar,abar}) - stepsizeP >=0 && P{sbar,abar}(j) - stepsizeP >=0
            %if P{sbar,abar}(j) - stepsizeP >=0
                P{sbar,abar}(j) = P{sbar,abar}(j) - stepsizeP;
                for a = 1: A{sbar}
                    if a~= abar && sum(P{sbar,a}) + stepsizeP <= 1 && P{sbar,a}(j) + stepsizeP <= 1
                    %if a~= abar && P{sbar,a}(j) + stepsizeP <= 1
                        P{sbar,a}(j) = P{sbar,a}(j) + stepsizeP;
                    end
                end
            end            
        end  
    end
    
    %Store old DV values in case next sol'n is worse
    P_iterminus1 =P;
    R_iterminus1= R;
    B_iterminus1=B;
    
    %Change the immediate reward values for both bifurcated sets
    R{sbar}(abar) =  R{sbar}(abar)+ stepsizeR;
    for a=1:A{sbar}
        if a~= abar
           R{sbar}(a) =  R{sbar}(a)- stepsizeR; 
        end
    end
    
    %Update new Z, U and A values via the Limited BI algorithm
    %[z(iter),U,A]= LimitedBI(R,P,B,T,tbar,sbar,abar, cardS,alpha,mu1,mu2,mu3,Rhat,Bhat, Phat,epsilon);
    [z(iter),U,A]= LimitedBISmooth(R,P,B,T,tbar,sbar,abar, cardS,alpha,mu1,mu2,mu3,Rhat,Bhat, Phat,epsilon);
    %Update the obj function relative change 
    zchange = ( z(iter-1)-z(iter) ) / abs(z(iter-1));
    else
        zchange = -inf;
    end  
end

%Record final values
z= z(~isnan(z)); %%Errorhandling step as discussed in Sec 3.1 of text
z_final = min(z); %Used instead of error handling logic
time = toc; %End timing
end


