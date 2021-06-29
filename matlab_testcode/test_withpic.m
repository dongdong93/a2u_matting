
clear;
trimapdir = '/PATH_TO/Combined_Dataset/Test_set/trimaps/';
alphadir = '/PATH_TO/Combined_Dataset/Test_set/alpha/';
preddir = '/PATH_TO/results/';
 
alphalist = dir(alphadir);
alphalist = alphalist(3:end);

loss_gradient = 0;
loss_sad = 0;
loss_mse = 0;
loss_conn = 0;
num = 0;
for i = 1:50
    for j =1:20
        num = num + 1;
        alphaname = alphalist(i).name;
        predname = strcat(alphaname(1:uint8(length(alphaname)-4)), '_', num2str(j-1), '.png');
        alpha = imread(strcat(preddir, predname));


        alpha_gt = imread(strcat(alphadir,alphaname));
        alpha_gt = alpha_gt(:,:,1);
        trimap = imread(strcat(trimapdir,predname));
        loss_g = compute_gradient_loss(alpha, alpha_gt, trimap);
        loss_gradient = loss_gradient + loss_g;
        loss_s = compute_sad_loss(alpha, alpha_gt, trimap);
        loss_sad = loss_sad + loss_s;
        loss_m = compute_mse_loss(alpha, alpha_gt, trimap);
        loss_mse = loss_mse + loss_m;
        loss_c = compute_connectivity_error(alpha, alpha_gt, trimap, 0.1);
        loss_conn = loss_conn + loss_c;
        fprintf('test: %d, SAD: %3.5f, MSE: %3.5f, Grad: %3.5f, Conn: %3.5f\n', num, loss_sad/num, loss_mse/num, loss_gradient/(num*1000), loss_conn/(num*1000))
    end
end
loss_gradient = loss_gradient/num;
loss_sad = loss_sad/num;
loss_mse = loss_mse/num;
loss_conn = loss_conn/num;
fprintf('test: %d, SAD: %3.5f, MSE: %3.5f, Grad: %3.5f\n, Conn: %3.5f\n', num, loss_sad, loss_mse, loss_gradient/1000, loss_conn/1000)
