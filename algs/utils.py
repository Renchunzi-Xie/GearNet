def create_alg(dataloader_s, dataloader_t, dataloader_test, device, args):
    if args.alg == 'gearnet_coteaching':
        from algs.gearnet_coteaching import GearNet_CoTeaching
        alg_obj = GearNet_CoTeaching(dataloader_s, dataloader_t, dataloader_test, device, args)
    elif args.alg == 'gearnet_dann':
        from algs.gearnet_dann import GearNet_DANN
        alg_obj = GearNet_DANN(dataloader_s, dataloader_t, dataloader_test, device, args)
    elif args.alg == 'gearnet_tcl':
        from algs.gearnet_tcl import GearNet_TCL
        alg_obj = GearNet_TCL(dataloader_s, dataloader_t, dataloader_test, device, args)
    return alg_obj