def download_blob(bucket_name, source_blob_name, destination_file_name):
    """_summary_

    Args:
        bucket_name (_type_): _description_
        source_blob_name (_type_): _description_
        destination_file_name (_type_): _description_
    """
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    print('Blob {} downloaded to {}.'.format(
    source_blob_name,
    destination_file_name))


def create_folder(bucket_name, destination_folder_name):
    """_summary_

    Args:
        bucket_name (_type_): _description_
        destination_folder_name (_type_): _description_
    """
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_folder_name)

    blob.upload_from_string('')

    print('Created {} .'.format(
        destination_folder_name))



def handler(event,context):
    """_summary_

    Args:
        event (_type_): _description_
        context (_type_): _description_
    """
    global model
    global today

    #the name of the new file that was uploaded in the bucket
    filename = event['name']
    input_bucket = event['bucket']#name of the bucket
    output_bucket = 'output-autoencoder'#output bucket


    #bucket paht 'gs//bucket_name/file_name
    input_path = 'gs://{0}/{1}'.format(input_bucket,filename)#input bcuket path
    #output_path = 'gs://{0}/{1}'.format(output_bucket,filename)#output bucket path

    #read the parquet file
    df = pd.read_parquet(input_path)
    arr = df.to_numpy()

    if (today is None) or (today != date.today()):
    today = date.today()
    folder = create_folder(output_bucket, '{0}/'.format(today))
    time = datetime.now().time()
    output_path = 'gs://{0}/{1}/{2}_{3}'.format(output_bucket,today,filename,time)#output bucket path
    # Model load which only happens during cold starts
    if model is None:
    download_blob('bucket-cred-score', 'autoencoder_weigths.index', '/tmp/autoencoder_weigths.index')
    download_blob('bucket-cred-score', 'autoencoder_weigths.data-00000-of-00001', '/tmp/autoencoder_weigths.data-00000-of-00001')
    model = Autoencoder(17,intermediary_layers=[300,200,100,50,25],z_layer=30)
    model.load_weights('/tmp/autoencoder_weigths')

    # Run prediction
    pred  = model.predict(arr)
    pred_loss = keras.losses.mae(pred,arr).numpy()
    df_output = pd.DataFrame(pred_loss,columns = ['pred_loss'])
    df_output.loc[df_output['pred_loss'] < 0.05 ,'pred_class'] = 'Normal'
    df_output.loc[df_output['pred_loss'] > 0.05 ,'pred_class'] = 'Anomalia'

    df_output.to_parquet(output_path)
