function [z_final,time, R, P, B, Rdot, Pdot, Bdot,U] = PoisonFiniteMDP_GradientDescent(T,tbar,sbar,abar, cardS,alpha,mu1,mu2,mu3,Rhat,Bhat, Phat,epsilon, stepsize, zchangelimit)

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
%stepsize: amount traveled on the gradient for each iteration
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

%Begin loop for whole algrithm -- loop until termination critera met
while zchange> zchangelimit
%For each state, reward-to-go at end set to terminal reward    
for s = 1:cardS
    U(T,s) = B{s}; 
end

%Limited BI Main Step -- Find action taken at tbar, sbar pair
for t=T-1:-1:tbar+1
    for s = 1:cardS
     for a =1:A{s}
        accumulator=0; %Used to iteratively cycle thru states & find summation
        for j=1:k-1
            accumulator(j) = P{s,a}(j)*(U(t+1,j)-U(t+1,k)); 
        end
        w{t,s}(a) = R{s}(a)+U(t+1,k)+sum(accumulator); %Candiate reward value for each state
     end
     %Find best rewarded action -- Use smooth max function (below) vice max()
     U(t,s) = exp(alpha*w{t,s}) * w{t,s}' / sum(exp(alpha*w{t,s})); 
    end 
end

%Find Preferred Action at time t
t=tbar;
s=sbar;
 for a =1:A{sbar}
    accumulator=0; %Accumulates terms in summation
    for j=1:k-1
        accumulator(j) = P{s,a}(j)*(U(t+1,j)-U(t+1,k));
    end
    w{t,s}(a) = R{s}(a)+U(t+1,k)+sum(accumulator);
 end
 %Find preffered action-- used softmax vs. argmax
 Astar = exp(alpha*w{t,s}(abar))/sum(exp(alpha*w{t,s}));
 
%Find Objective Function
for i=1:size(Rhat,2)
    %Finds terms in the summation of each DV variable change
    diffVecb(i) = sum((Bhat{i}-B{i}).^2 / (Bhat{i}.^2 + epsilon) );
    diffvecR(i) = sum((Rhat{i} - R{i}).^2 / (Rhat{i}.^2 + epsilon) );
    for j =1:A{i}
        diffMatP(i,j) = sum((Phat{i,j}-P{i,j}).^2 / (Phat{i,j}.^2 + epsilon));
    end
end
%Calculates components of the obj function
d1 = sum(diffvecR);
d2 = sum(sum(diffMatP));
d3 = sum(diffVecb);
%Calculates actual obj func
z(iter)= -Astar +mu1*d1+mu2*d2+mu3*d3;

% %Absolute Change -- Check if zchange is satisied and leave loop if req'd
% if iter>1
%     zchange = z(iter)-z(iter-1);
%     if zchange>= zchangelimit
%         break
%     end 
% end

% %Relative Change -- Check if zchange is satisied and leave loop if req'd
 if iter>1
     zchange = ( z(iter-1)-z(iter) ) / abs(z(iter-1));
     if zchange<= zchangelimit
         break
     end 
 end


%Find Gradient Direction -- Implements autodiff found by hand
for a =1:A{sbar}
    if a ~= abar
        wdot{tbar,sbar}(a)=-1*Astar*(-exp(alpha*w{tbar,sbar}(a))/sum(exp(alpha*w{tbar,sbar})));
    else
        wdot{tbar,sbar}(a)=-1*Astar*(1-exp(alpha*w{tbar,sbar}(a))/sum(exp(alpha*w{tbar,sbar})));
    end 
end

Udot = zeros(T,cardS);
for s=1:cardS
    Bdot{s} = mu3 *(2*B{s}-2*Bhat{s})/(Bhat{s}.^2 + epsilon);
    if s~=k
        for a=1:A{s}
            Udot(tbar+1,s) = Udot(tbar+1,s) + wdot{tbar,sbar}(a)*P{sbar,a}(s);
        end
    else
        for a =1:A{s}
          calcp_k=0;
          for j=1:k-1
            calcp_k=calcp_k + P{sbar,a}(j);   %To do sum product in below equation
          end
        Udot(tbar+1,s) = Udot(tbar+1,s) + wdot{tbar,sbar}(a)*(1-calcp_k);
        end 
    end
    for a =1:A{s}
        Rdot{s}(a) = mu1 *(2*R{s}(a)-2*Rhat{s}(a))/(Rhat{s}(a).^2 + epsilon) ;
        if s== sbar
            Rdot{s}(a) = Rdot{s}(a) + wdot{tbar,sbar}(a);
        end
        for j =1:k-1
            Pdot{s,a}(j) = mu2 *(2*P{s,a}(j)-2*Phat{s,a}(j))/ (Phat{s,a}(j).^2 + epsilon);
            if s==sbar
                Pdot{s,a}(j) = Pdot{s,a}(j) + wdot{tbar,sbar}(a)*(U(tbar+1,j)-U(tbar+1,k));
            end
        end
    end
end

%Continue gradient search into recursive loop
%%Bdot, Udot and Rdot are the gradient direction vectors for B, U and R
i=1;
for t= tbar+1:T-1
    for s=1:cardS
        for a =1:A{s}
            softmax =exp(alpha*w{t,s}(a))/sum(exp(alpha*w{t,s}));
            wdot{t,s}(a)=Udot(t,s)*softmax*(1+alpha*(w{t,s}(a)-U(t,s)));
            Rdot{s}(a) = Rdot{s}(a) + wdot{t,s}(a);
            for j=1:k
                if j ~= k
                 Pdot{s,a}(j) = Pdot{s,a}(j) + wdot{t,s}(a)*(U(t+1,j)-U(t+1,k));
                end
                if t< T-1 && j~= k
                    Udot(t+1,j) = Udot(t+1,j)+ wdot{t,s}(a)*P{s,a}(j);
                elseif t< T-1 && j==k
                    Udot(t+1,j) = Udot(t+1,j)+ wdot{t,s}(a)*(1-sum(P{s,a}));
                elseif t== T-1 && j~=k
                    Bdot{j} =  Bdot{j}+ wdot{t,s}(a)*P{s,a}(j);
                else
                    Bdot{j} =  Bdot{j}+wdot{t,s}(a)*(1-sum(P{s,a}));
                end
            end
               
        end
    end
end

iter = iter+1;

%Update new R, B and P values
if isnan(z(iter-1)) ~=1  
 for s=1:cardS
    B{s} = B{s} - stepsize*Bdot{s}; %Move in direction IAW stepsize
    for a =1:A{s}
        R{s}(a) = R{s}(a) - stepsize*Rdot{s}(a); %Move in direction IAW stepsize
        for j=1:k-1
           test = P{s,a}(j) - stepsize*Pdot{s,a}(j); %Move in direction IAW stepsize
           checksum = sum(P{s,a})-P{s,a}(j)+test;
           if checksum>=0 && checksum<=1  %Move in improving direction w/o breaking constraints
              P{s,a}(j) = test;
           end
        end
    end
 end
end

end 

%For export
%If error found based on softmax, report last good found value
z= z(~isnan(z)); %Errorhandling step as discussed in Sec 3.1 of text
z_final = z(length(z));
time = toc; %End timing

end
