def debug(address=None, breakpoint_=True, rank=None):
    import logging
    import os
    import torch
    logger = logging.getLogger(__name__)
    address_source = 'keyword argument'
    if address is None:
        address = os.environ.get('DEBUG_ADDRESS')
        address_source = 'environment variable (DEBUG_ADDRESS)'
    if address:
        try:
            host, port = address.split(':')
            port = int(port)
        except ValueError:
            logger.error(f'Invalid debug address, got {address} from {address_source} but format should be "HOST:PORT"')
            print("================= Value Error =====================")
            return
        try:
            if rank is None:
                rank = 0
            port += rank
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == rank:
                import debugpy
                additional_msg = ' with breakpoint' if breakpoint_ else ''
                logger.info(f'Attempting to connect to debug server at {host}:{port}{additional_msg}')
                debugpy.connect((host, port))
                if breakpoint_:
                    debugpy.breakpoint()
            else:
                logger.info(
                    f'Skipped debug connection, wrong rank ({torch.distributed.get_rank()}) only connecting from rank={rank}')

                print("------------------------")
                print("is init: ", torch.distributed.is_initialized())
                print("rank: ", torch.distributed.get_rank(), " ", rank)
                print("+++++++++++++++++++++++ Not Distributed ++++++++++++++++++++++")
        except Exception as e:
            logger.error(e)
            print("------------------------ LAST -----------------------")