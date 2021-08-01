%% Cargamos los datos
% Descomprimimos las imagenes y las guardamos como un almacén de datos de imágenes 
% llamada |imageDatastore|
% Etiquetamos las imagenes según el nombre de sus carpetas
unzip('MerchData.zip');
images = imageDatastore('MerchData',...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');

%%
% Dividimos los datos de entrenamiento y validación. Utilizamos el 70 % de
% las imagenes para entrenamiento y un 30 % para validación -
% "splitEachLabel" permite dividir las imagenes en dos almacenes nuevos 
[trainingImages,validationImages] = splitEachLabel(images,0.7,'randomized');

%%
% Ahora tenemos 55 imágenes de entrenamiento y 20 de validación, mostramos
% unas imagenes a modo ejemplo
numTrainImages = numel(trainingImages.Labels);
idx = randperm(numTrainImages,16);
figure
for i = 1:16
    subplot(4,4,i)
    I = readimage(trainingImages,idx(i));
    imshow(I)
end

%% Cargamos la red preentrenada Alexnet

net = alexnet;

%% Mostramos la arquitectura de la red. Posee 5 capas convuolcionales y 
%% 3 capas completamente conectadas
net.Layers



%% Transferimos las capas a la nueva red
% Las últimas 3 capas de la red preentrenada "net" están configuradas para
% 1000 clases, estas 3 capas deben ajustarse para el nuevo problema de
% clasificación, extraemos todas las capas, exceptuando las últimas 3 de la
% red preentrenada
layersTransfer = net.Layers(1:end-3);

%%
% Transferimos las capas a la nueva tarea de clasificación reemplazando las
% últimas 3 capas por:una capa completamente conectada, una capa softmax y
% una capa de salida de clasificación. Establecemos el mismo tamaño de
% clases para los datos
numClasses = numel(categories(trainingImages.Labels))
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];



%% Opciones de entrenamiento
% Especificamos las opciones de entrenamiento, para realizar la trasnferen
% de aprendizaje, mantenemos las funciones de las primeras capas de la reed
% preentrenada, establecemos InitialLearnRate a un valor pequeño para
% relantizar el aprendizaje en las capas transferidas, en el paso anterior,
% se aumentó la tasa de aprendizaje para la capa completamente conectada
% para acelerar el aprendizaje en las capas finales, esta combinación de
% tasas de aprendizaje da como resultado un aprendizaje rápido solo en las
% nuevas capas y un aprendizaje más lento en las otras.
miniBatchSize = 10;
numIterationsPerEpoch = floor(numel(trainingImages.Labels)/miniBatchSize);
options = trainingOptions('sgdm',...
    'MiniBatchSize',miniBatchSize,...
    'MaxEpochs',4,...
    'InitialLearnRate',1e-4,...
    'Verbose',false,...
    'Plots','training-progress',...
    'ValidationData',validationImages,...
    'ValidationFrequency',numIterationsPerEpoch);   %% Detiene el ent si la pérdida de validación deja de mejorar

%% Al realizar un aprendizaje por transferencia, no es necesario entrenar 
% durante tantas épocas. Una época es un ciclo de entrenamiento completo en
% todo el conjunto de datos de entrenamiento, Especificamos el tamaño del
% mini lote y los datos de validación


%%
% Entrenamos la red
netTransfer = trainNetwork(trainingImages,layers,options);


%% Clasificamos las imágenes de validación
predictedLabels = classify(netTransfer,validationImages);

%%
% Mostramos las 4 imágenes de validación con sus etiquetas
idx = [1 5 10 15];
figure
for i = 1:numel(idx)
    subplot(2,2,i)
    I = readimage(validationImages,idx(i));
    label = predictedLabels(idx(i));
    imshow(I)
    title(char(label))
end

%%
% Calculamos la precisión de la clasificación en el conjunto de validación.
% La precisión es la fracción de etiquetas que la red predice correctamente
valLabels = validationImages.Labels;
accuracy = mean(predictedLabels == valLabels)

