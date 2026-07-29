cpu_family = host_machine.cpu_family()
system = host_machine.system()

if cpu_family == 'x86_64'
    subdir('x86-64')
elif cpu_family == 'aarch64' and system == 'darwin'
    subdir('aarch64/macos')
else
    error('Unsupported platform: ' + system + '/' + cpu_family)
endif
