clc;
clear;
close all;

%% PARAMETERS (Cấu hình tham số của riêng ông)
videoFile = "C:\Users\Admin\Downloads\file_example_MP4_640_3MG.mp4";
QP = 50;              % Thử QP = 15 (nét) và QP = 40 (xuất hiện rõ lỗi khối Artifacts)
blockSize = 32;         % Kích thước khối biến đổi (Transform Size)
maxFrames = 100;       % Giới hạn 100 frames để máy chạy nhanh, ra đồ thị ngay

%% Read video
v = VideoReader(videoFile);
outputName = ['recon_B' num2str(blockSize) '_QP' num2str(QP) '.mp4'];

% Khởi tạo bộ ghi video đầu ra
writer = VideoWriter(outputName, 'MPEG-4');
writer.Quality = max(10, 100 - QP*2); 
writer.FrameRate = v.FrameRate; 
open(writer);

% Khởi tạo các mảng lưu số liệu (Expected Output của thầy)
psnrValues = [];
mseValues = [];
avgResidualErrors = []; 

frameCount = 0;
Qstep = 2^((QP-4)/6);
previousFrame = [];

fprintf('Hệ thống đang xử lý luồng Video... Đợi vài giây có đồ thị ngay bro...\n');

%% Processing Loop (Stream từng frame, RAM siêu nhẹ, không lo treo máy)
while hasFrame(v) && frameCount < maxFrames
    frameCount = frameCount + 1;
    frame = double(readFrame(v));
    [H, W, C] = size(frame);
    
    % Cắt biên để đảm bảo ma trận chia hết cho blockSize
    H_clean = H - mod(H, blockSize);
    W_clean = W - mod(W, blockSize);
    frame = frame(1:H_clean, 1:W_clean, :);
    
    %% Prediction Stage (Ước lượng phần dư)
    if isempty(previousFrame)
        predicted = frame;
    else
        predicted = previousFrame;
    end
    
    reconFrame = zeros(H_clean, W_clean, C);
    frame_residual_sum = 0;
    
    %% Process RGB channels
    for c = 1:C
        residual = frame(:,:,c) - predicted(:,:,c);
        reconResidual = zeros(H_clean, W_clean);
        
        % Tích lũy trị tuyệt đối phần dư phục vụ thống kê (Residual Stats)
        frame_residual_sum = frame_residual_sum + mean(abs(residual(:)));
        
        %% Block processing (Biến đổi & Lượng tử hóa từng khối)
        for row = 1:blockSize:H_clean-blockSize+1
            for col = 1:blockSize:W_clean-blockSize+1
                block = residual(row:row+blockSize-1, col:col+blockSize-1);
                
                % Lõi thuật toán: DCT -> Quantization -> Inverse Quantization -> IDCT
                coeff = dct2(block);
                Qcoeff = round(coeff / Qstep);
                reconCoeff = Qcoeff * Qstep;
                reconResidual(row:row+blockSize-1, col:col+blockSize-1) = idct2(reconCoeff);
            end
        end
        % Tái tạo lại kênh màu bằng cách cộng bù sai số vào frame dự đoán
        reconFrame(:,:,c) = predicted(:,:,c) + reconResidual;
    end
    
    % Giới hạn vùng giá trị điểm ảnh hợp lệ [0, 255]
    reconFrame = max(0, min(255, reconFrame));
    
    %% Thống kê số liệu đầu ra (Expected Output)
    % 1. Tính toán Distortion Metrics (PSNR & MSE)
    p = psnr(uint8(reconFrame), uint8(frame));
    psnrValues(end+1) = p;
    
    m = mean((frame(:) - reconFrame(:)).^2);
    mseValues(end+1) = m;
    
    % 2. Lưu lại Residual Stats của frame hiện tại
    avgResidualErrors(end+1) = frame_residual_sum / C;
    
    %% Ghi khung hình vào video đầu ra
    writeVideo(writer, uint8(reconFrame));
    
    %% Cập nhật bộ đệm dự đoán cho khung hình tiếp theo
    previousFrame = reconFrame;
end
close(writer);

%% ==================== AUTOMATIC PLOTTING & AUTO-SAVE ====================
% Tạo figure ẩn danh đặt tên theo thông số để phân biệt
figTitle = ['Metrics_Plot_B' num2str(blockSize) '_QP' num2str(QP)];
hFig = figure('Name', figTitle, 'Position', [100, 100, 1200, 400]);

% Đồ thị 1: Distortion Metric (PSNR)
subplot(1, 3, 1);
plot(psnrValues, 'LineWidth', 2, 'Color', 'b');
xlabel('Khung hình (Frame)', 'FontWeight', 'bold'); 
ylabel('PSNR (dB)', 'FontWeight', 'bold');
title(['Distortion Metric (PSNR) - QP=' num2str(QP)]); 
grid on;

% Đồ thị 2: Distortion Metric (MSE)
subplot(1, 3, 2);
plot(mseValues, 'LineWidth', 2, 'Color', 'r');
xlabel('Khung hình (Frame)', 'FontWeight', 'bold'); 
ylabel('Sai số (MSE)', 'FontWeight', 'bold');
title(['Distortion Metric (MSE) - B=' num2str(blockSize)]); 
grid on;

% Đồ thị 3: Residual Stats (Sai số phần dư trung bình)
subplot(1, 3, 3);
plot(avgResidualErrors, 'LineWidth', 2, 'Color', 'g');
xlabel('Khung hình (Frame)', 'FontWeight', 'bold'); 
ylabel('Avg Residual Error', 'FontWeight', 'bold');
title('Residual Stats'); 
grid on;

%% FIX CỐT LÕI: Tự động lưu 3 đồ thị thành file ảnh PNG không bao giờ lo mất
imageOutputName = [figTitle '.png'];
saveas(hFig, imageOutputName);
fprintf('📸 Đã lưu thành công ảnh đồ thị: %s\n', imageOutputName);

%% In bảng kết quả tóm tắt ra Command Window
fprintf('\n================== KẾT QUẢ THỰC NGHIỆM TỔNG HỢP ==================\n');
fprintf('Kích thước khối (Transform Size) = %d x %d\n', blockSize, blockSize);
fprintf('Tham số lượng tử (QP) = %d\n', QP);
fprintf('------------------------------------------------------------------\n');
fprintf('1. RESIDUAL STATS:\n');
fprintf('   - Sai số phần dư trung bình tổng thể: %.4f\n', mean(avgResidualErrors));
fprintf('2. DISTORTION METRICS:\n');
fprintf('   - Chất lượng khách quan trung bình (PSNR): %.2f dB\n', mean(psnrValues));
fprintf('   - Độ méo bình phương trung bình (MSE): %.4f\n', mean(mseValues));
fileInfo = dir(outputName);
fprintf('Dung lượng tệp đầu ra: %.2f KB\n', fileInfo.bytes / 1024);

% Tự động bật video đầu ra để kiểm tra lỗi Artifacts trực quan
winopen(outputName);