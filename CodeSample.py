def unetpp(size_of_image=(625, 625, 5)):
    sampleinput = Input(size_of_image)
    one = unetpp_block(sampleinput, 64)
    second = MaxPooling2D(pool_size=(2, 2))(one)

    up = unetpp_block(second, 128)
    down = MaxPooling2D(pool_size=(2, 2))(up)

    d_01 = Conv2DTranspose(64, kernel_size=(2, 2), strides=(2, 2), padding='same')(up)
    d_01_c = concatenate([one, d_01])
    d_01_f = unetpp_block(d_01_c, 64)

    # Second Row
    down3 = unetpp_block(down, 256)
    pool3 = MaxPooling2D(pool_size=(2, 2))(down3)

    s2 = Conv2DTranspose(64, kernel_size=(2, 2), strides=(2, 2), padding='same')(down3)
    s2_c = concatenate([up, s2])
    s2_f = unetpp_block(s2_c, 64)

    s3 = Conv2DTranspose(64, kernel_size=(2, 2), strides=(2, 2), padding='same')(s2_f)
    s3_c = concatenate([one, d_01_f, s3])
    s3_f = unetpp_block(s3_c, 64)

    # Third Row
    down4 = unetpp_block(pool3, 512, True)
    pool4 = MaxPooling2D(pool_size=(2, 2))(down4)

    s4 = Conv2DTranspose(64, kernel_size=(2, 2), strides=(2, 2), padding='same')(down4)
    s4_c = concatenate([down3, s4])
    s4_f = unetpp_block(s4_c, 64)

    s5 = Conv2DTranspose(64, kernel_size=(2, 2), strides=(2, 2), padding='same')(s4_f)
    s5_c = concatenate([up, s2_f, s5])
    s5_f = unetpp_block(s5_c, 64)

    s6 = Conv2DTranspose(64, kernel_size=(2, 2), strides=(2, 2), padding='same')(s5_f)
    s6_c = concatenate([one, d_01_f, s3_f, s6])
    s6_f = unetpp_block(s6_c, 64)

    # Bottleneck Layer
    down5 = unetpp_block(pool4, 1024, True)

    # Upsampling Layer 2
    s7 = Conv2DTranspose(512, kernel_size=(2, 2), strides=(2, 2), padding='same')(down5)
    s7_c = concatenate([s7, down4])
    s7_f = unetpp_block(s7_c, 512)

    # Upsampling Layer 2
    s8 = Conv2DTranspose(256, kernel_size=(2, 2), strides=(2, 2), padding='same')(s7_f)
    s8_c = concatenate([s8, down3, s4_f])
    s8_f = unetpp_block(s8_c, 256)

    # Upsampling Layer 3
    s9 = Conv2DTranspose(128, kernel_size=(2, 2), strides=(2, 2), padding='same')(s8_f)
    s9_c = concatenate([s9, up, s2_f, s5_f])
    s9_f = unetpp_block(s9_c, 128)

    # Upsampling Layer 4
    d_010 = Conv2DTranspose(64, kernel_size=(2, 2), strides=(2, 2), padding='same')(s9_f)
    d_010_c = concatenate([d_010, one, d_01_f, s3_f, s6_f])
    d_010_f = unetpp_block(d_010_c, 64)

    convOut = Conv2D(3, 1, activation='sigmoid')(d_010_f)

    model = Model(input=sampleinput, output=convOut)

    return model