% Program to perform GCMC, G-GCMC and C-GCMC and draw error bars vs time

clc
clear all
close all


% for K=40
K=40;  % number of workers
groups=[40 20 10]; % number of groups in 3 cases
redundancy=[1 2 4]; % redundancy r in 3 cases  


d=5; % dimension of posterior distribution

L_max = 30; % no of samples generated in each realization
Z=50;      % no of realizations for plotting error bars
sigma_sq1 = 10;  % regularization parameter
sigma_sq2 = 2; % regularization parameter 
std_coeff = 0.15; % to control standard deviation in error bars

eta = 0.1;  % mean for exponential distribution and a paratmer in pareto distribution
beta = 1.2; % a paratmer in pareto distribution

mean_local = zeros(d,K); % considering all subposteriors have zero mean vector
mu_true_global = zeros(d,1); % consider global posterior has zero mean vector
rho = zeros(1,K); % used to define covariance matrices of K subposteriors
Cov_local = zeros(d,d,K);  % covariance matrices for K local subposterior
Cov_true_global = zeros(d,d); % covariance matrix for true global posterior




%%%%%%%% Defining covariance matrices of subposteriors, 
%%%%%%%% finding covariance matrix for the true global posterior and
%%%%%%%% computing f for true global posterior to use in error formula

for s=1:K
    mean_local(:,s)=zeros(d,1);
    
    rho(s) = (s-1)/K;
    Cov_local(:,:,s)= toeplitz((rho(s)*ones(1,d)).^[0:d-1]);
    
    Cov_true_global=Cov_true_global+inv(Cov_local(:,:,s));
    
end
Cov_true_global = inv(Cov_true_global); % covariance matrix of the true global posterior.


L_true_global= 10^6;  % # samples to be taken from the true global posterior.
% rng('default') % specifies the seed for the MATLAB random number generator

%generating true global samples
theta_true_global = transpose(mvnrnd(mu_true_global,Cov_true_global,L_true_global));


%computing f for true global samples to use in error function
f_true_global = zeros(d,d);

for l=1:L_true_global
    f_true_global = f_true_global + ( theta_true_global(:,l) ) * transpose( ( theta_true_global(:,l) ) );    
end

f_true_global = (1/L_true_global) * f_true_global ;  % this is needed in error formula



figure(1)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for x=1:3  % for 3 cases



r=redundancy(x);  % no of workers in a group = redundancy
G=groups(x);  % no of groups = no of shards


%%%%%%%%%%  COMPUTING TIME  %%%%%%%%%%

%%%%  Defining pareto distributions.

clear Delta_T pareto_dist 

% For Pareto distibuted RV
pareto_dist = makedist('GeneralizedPareto','k',(1/beta),'sigma',((eta*r*(beta-1))/(beta^2)),'theta',((eta*r*(beta-1))/beta));
Delta_T= random(pareto_dist,L_max,K,Z);




%%% Computing time for C-GCMC

clear cumsumDelta_T_CGCMC 
clear sorted_cumsumDelta_T_CGCMC
cumsumDelta_T_CGCMC = zeros(L_max,K,Z);
sorted_cumsumDelta_T_CGCMC = zeros(L_max,K,Z);

for z=1:Z
    cumsumDelta_T_CGCMC(:,:,z) = cumsum(Delta_T(:,:,z),1);
    sorted_cumsumDelta_T_CGCMC(:,:,z)= sort(cumsumDelta_T_CGCMC(:,:,z),2); % sorts each row in ascending order
end

clear T_CGCMC
T_CGCMC=zeros(L_max,Z);
for z=1:Z
    T_CGCMC(:,z) = sorted_cumsumDelta_T_CGCMC(:,K-r+1,z); % Time at which each sample is generated at server
end

clear mean_Time_CGCMC 
clear mean_Time_flip_CGCMC
mean_Time_CGCMC = mean( transpose(T_CGCMC(1:L_max,:)));
mean_Time_flip_CGCMC = [mean_Time_CGCMC fliplr(mean_Time_CGCMC)];  % To plot error bars

%%%%%%%%%%%%%%%%%%%%%%%%



%%% Computing time for G-GCMC

clear DELTA_T_GGCMC

% splitting workers into groups of size r  
for g=1:G
    DELTA_T_GGCMC{g} = Delta_T(:, ((g-1)*r+1):g*r, :); % each array is of size (L_max,r,Z)
end

clear cumsumDELTA_T_GGCMC
clear cumsumDelta_T_GGCMC
cumsumDelta_T_GGCMC = zeros(L_max*r,G,Z);

clear T_GGCMC
T_GGCMC=zeros(L_max*r,Z); 
% for L_max*r batches of samples generated at each worker or L_max*r global samples generated at the server

for z=1:Z
    for g=1:G
   
        cumsumDELTA_T_GGCMC{g}(:,:,z) = cumsum(DELTA_T_GGCMC{g}(:,:,z),1); 
        cumsumDelta_T_GGCMC(:,g,z) = sort(reshape(cumsumDELTA_T_GGCMC{g}(:,:,z),[],1 ));
        % ascending order of times at which batches of samples is received at server from group g

    end
    T_GGCMC(:,z) = max(cumsumDelta_T_GGCMC(:,:,z),[],2); 
end

clear mean_Time_GGCMC
clear mean_Time_flip_GGCMC
mean_Time_GGCMC = mean( transpose(T_GGCMC(1:L_max,:))); % we only consider L_max samples among L_max*r for ploting error bars
mean_Time_flip_GGCMC = [mean_Time_GGCMC fliplr(mean_Time_GGCMC)];


%%%%%%%%%%  END OF COMPUTING TIME  %%%%%%%%%%


clear theta phi_CGCMC mu C phi_mu_CGCMC D_CGCMC THETA_CGCMC error_CGCMC f_global_CGCMC

theta=zeros(d,L_max,K);
phi_CGCMC=zeros(d,L_max);
mu=zeros(d,K);
C=zeros(d,d,K);
phi_mu_CGCMC= zeros(d,L_max);
D_CGCMC = zeros(d,d,L_max);
THETA_CGCMC=zeros(d,L_max);
error_CGCMC=zeros(L_max,Z);


clear error_GGCMC_UpdateAll THETA_GGCMC_UpdateAll

error_GGCMC_UpdateAll=zeros(L_max,Z);
THETA_GGCMC_UpdateAll=zeros(d,L_max);


clear mean_error_CGCMC std_error_CGCMC CGCMC_curve1 CGCMC_curve2 CGCMC_curve_inBetween

clear mean_error_GGCMC_UpdateAll std_error_GGCMC_UpdateAll GGCMC_UpdateAll_curve1
clear GGCMC_UpdateAll_curve2 GGCMC_UpdateAll_inBetween




%%%%%%%%  Computing local samples and aggregating in C-GCMC and G-GCMC  %%%%%%%%%%%%%%%

for z=1:Z

    %%%%%% COMPUTING LOCAL SAMPLES for z-th realization

    for s=1:K
        theta(:,:,s) = transpose(mvnrnd(mean_local(:,s),Cov_local(:,:,s),L_max));
    end

    %%%%%%% findng updated covariances matrices for each sample generated
    
    for L=1:L_max
     
        for s=1:K
       
            mu(:,s) = zeros(d,1);
        
            C(:,:,s) = zeros(d,d);
        
            for l=1:L
                mu(:,s)=mu(:,s)+theta(:,l,s);
            end
        
            mu(:,s)=(1/L)*mu(:,s);
        
            for l=1:L
                C(:,:,s) = C(:,:,s)+ ( theta(:,l,s)-mu(:,s) )*transpose( theta(:,l,s)-mu(:,s) );
            end
        
            C(:,:,s) = sigma_sq1*eye(d,d) + (1/L)*C(:,:,s);
        
        end
        

        %%%%% C-GCMC calculations  %%%%%% 

        %computing sum of processed matrices
    
        phi_CGCMC(:,L) = zeros(d,1);
    
        for s=1:K
            phi_CGCMC(:,L) = phi_CGCMC(:,L) + (inv(C(:,:,s)))*theta(:,L,s);
        end

            
        % estimating sum of precision matrices from phi_CGCMC(:,l) for $l \in [L]$

        phi_mu_CGCMC(:,L) = zeros(d,1);
        D_CGCMC(:,:,L) = zeros(d,d);  % covariance matrix of phi_CGCMC(:,l) for $l \in [L]$

        for l=1:L
            phi_mu_CGCMC(:,L)=phi_mu_CGCMC(:,L)+phi_CGCMC(:,l);
        end
        
        phi_mu_CGCMC(:,L)=(1/L)*phi_mu_CGCMC(:,L);
        
        for l=1:L
            D_CGCMC(:,:,L) = D_CGCMC(:,:,L) + ( phi_CGCMC(:,l)-phi_mu_CGCMC(:,L) )*transpose( phi_CGCMC(:,l)-phi_mu_CGCMC(:,L) );
        end
        
        D_CGCMC(:,:,L) = sigma_sq2*eye(d,d) + (1/L)*D_CGCMC(:,:,L);
    

        
        % computing the L-th global sample at time T(L,z) 
                
        THETA_CGCMC(:,L)= inv(D_CGCMC(:,:,L))*phi_CGCMC(:,L) ;
    
        %computing f for produced L global samples to use in error function
        f_global_CGCMC = zeros(d,d);

        for l=1:L
            f_global_CGCMC = f_global_CGCMC + ( THETA_CGCMC(:,l) ) * transpose( ( THETA_CGCMC(:,l) ) );    
        end
    
        f_global_CGCMC = (1/L) * f_global_CGCMC ;
    
        %finding error upto T(L)
    
        error_CGCMC(L,z) = 0;
    
        for i=1:d
            for j=1:d
                error_CGCMC(L,z) = error_CGCMC(L,z) + ( abs(f_global_CGCMC(i,j) - f_true_global(i,j)) / f_true_global(i,j) ) ;
            end
        end 
    
        error_CGCMC(L,z) = (1/(d^2)) * error_CGCMC(L,z);
              
        %%%%% end of C-GCMC calculations  %%%%%%



        %%%%% G-GCMC calculations (for UpdteAll protocol)  %%%%%%

        %computing weight matrix for aggregation
    
        Weight = zeros(d,d);
    
        for s=1:K
            Weight = Weight + inv(C(:,:,s));
        end

        Weight = inv(Weight);

        
        %computing all global samples produced at every time T(L) (time at which L^{th} sample is received)
        
         
        for l=1:L
            THETA_GGCMC_UpdateAll(:,l) = zeros(d,1);
            for s=1:K
                THETA_GGCMC_UpdateAll(:,l) =  THETA_GGCMC_UpdateAll(:,l) + Weight*inv(C(:,:,s))*theta(:,l,s);
            end
        end
    
    
        %computing f for produced L global samples to use in error function
        f_global_GGCMC_UpdateAll = zeros(d,d);

        for l=1:L
            f_global_GGCMC_UpdateAll = f_global_GGCMC_UpdateAll + ( THETA_GGCMC_UpdateAll(:,l) ) * transpose( ( THETA_GGCMC_UpdateAll(:,l) ) );    
        end
    
        f_global_GGCMC_UpdateAll = (1/L) * f_global_GGCMC_UpdateAll ;
    
        %finding error upto T(L)
    
        error_GGCMC_UpdateAll(L,z) = 0;
    
        for i=1:d
            for j=1:d
                error_GGCMC_UpdateAll(L,z) = error_GGCMC_UpdateAll(L,z) + ( abs(f_global_GGCMC_UpdateAll(i,j) - f_true_global(i,j)) / f_true_global(i,j) ) ;
            end
        end 
    
        error_GGCMC_UpdateAll(L,z) = (1/(d^2)) * error_GGCMC_UpdateAll(L,z);

    end
    
end


%%%%%%%%%%% Error bars and averges for C-GCMC

if r ~= 1
mean_error_CGCMC = mean( transpose(error_CGCMC) ) ;
std_error_CGCMC = std( transpose(error_CGCMC) ) ;

CGCMC_curve1 = mean_error_CGCMC + std_coeff*std_error_CGCMC;
CGCMC_curve2 = mean_error_CGCMC - std_coeff*std_error_CGCMC;

CGCMC_curve_inBetween = [CGCMC_curve1, fliplr(CGCMC_curve2)];

%%%%

plot( mean_Time_CGCMC, mean_error_CGCMC,'--', 'Linewidth',2,'Color', [0.4660, 0.6740, 0.1880] )
hold on


fill( mean_Time_flip_CGCMC, CGCMC_curve_inBetween, 1,'facecolor', [0.4660, 0.6740, 0.1880], 'edgecolor', 'none', 'facealpha', 0.2);
hold on

end

%%%%%%%%%%% Error bars and averges for G-GCMC


mean_error_GGCMC_UpdateAll = mean( transpose(error_GGCMC_UpdateAll) ) ;
std_error_GGCMC_UpdateAll = std( transpose(error_GGCMC_UpdateAll) ) ;

GGCMC_UpdateAll_curve1 = mean_error_GGCMC_UpdateAll + std_coeff*std_error_GGCMC_UpdateAll;
GGCMC_UpdateAll_curve2 = mean_error_GGCMC_UpdateAll - std_coeff*std_error_GGCMC_UpdateAll;

GGCMC_UpdateAll_inBetween = [GGCMC_UpdateAll_curve1, fliplr(GGCMC_UpdateAll_curve2)];

%%%%
if r==1
    plot( mean_Time_GGCMC, mean_error_GGCMC_UpdateAll,'-o', 'Linewidth',2,'Color', [0.4940, 0.1840, 0.5560]	 )
    hold on

    fill( mean_Time_flip_GGCMC, GGCMC_UpdateAll_inBetween,  1,'facecolor',[0.4940, 0.1840, 0.5560], 'edgecolor', 'none', 'facealpha', 0.2);
    hold on
    %this will be plotted first as r=1

else
    plot( mean_Time_GGCMC, mean_error_GGCMC_UpdateAll,'-', 'Linewidth',2,'Color', [ 0.8500    0.3250    0.0980] )
    hold on
    
    fill( mean_Time_flip_GGCMC, GGCMC_UpdateAll_inBetween,  1,'facecolor',[ 0.8500    0.3250    0.0980] , 'edgecolor', 'none', 'facealpha', 0.2);
    hold on
end    

end


hold off

legend('CMC','','C-CMC','','G-CMC','Location','northeast')


xlabel('Time ','fontsize',20)
ylabel('Test Error','fontsize',20)
set(gca,'FontName','Times New Roman','FontSize',20);

% grid on

% set(gcf, 'Position',  [100, 100, 400, 400])
axis square



xlim([0,3.2])
ylim([0.5,5])








