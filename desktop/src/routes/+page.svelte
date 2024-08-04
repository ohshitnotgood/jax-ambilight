<script lang="ts">
    import { invoke } from "@tauri-apps/api/tauri";
    import * as utils from "../components/utils";
    import { Command } from "@tauri-apps/api/shell";
    import { appWindow } from "@tauri-apps/api/window";

    const availableHeightZones = [2, 3, 4, 5, 6];
    const availableWidthZones = [4, 5, 6, 7, 8];

    let nWidthZones = 6;
    let nHeightZones = 4;
    let monitor = "Select monitor";

    $: colourList = [[""]];
    $: gradientList = [""];
    $: isDataReady = false;
    $: useGradientPreview = false;

    async function startPreview() {
        setInterval(async () => {
            let ipcResponse: string = await invoke(
                "write_and_wait_for_response_blocking",
                { message: "ack_ok" },
            );
            ipcResponse = ipcResponse.replaceAll("'", "");
            ipcResponse.slice(1);
            colourList = utils.parseBackendData(ipcResponse);
            gradientList = utils.parseBackendDataAndCSSGradient(ipcResponse);
            isDataReady = true;
        }, 100);
    }


    async function onZoneNumberChange() {
        // await invoke(
        //     "write_and_wait_for_response_blocking",
        //     { message: `chg_v:${nHeightZones};${nWidthZones}`},
        // );
    }

    async function testRunningCommands() {
        console.log("Working")
        const command = new Command("run-touch", "/home/praanto/file.file")
        command.stdout.on("data", line => console.log(line))
        await command.execute()
    }
</script>

<main class="grid place-content-center w-screen h-screen">
    {#if !isDataReady}
        <connect class="block text-center">
            <button class="text-white bg-blue-700 hover:bg-blue-800 focus:ring-4 focus:ring-blue-300 font-medium rounded-lg text-sm px-5 py-2.5 me-2 mb-2" on:click={startPreview}> 
                Connect to backend 
            </button>
        </connect>
    {/if}

    {#if isDataReady}
        <!-- <controls class="grid grid-cols-2 gap-x-5 gap-y-2 text-right">
            <div>Width zones</div>
            <div>
                <select on:change={onZoneNumberChange}
                    bind:value={nWidthZones}
                    class="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-md block px-4 py-1"
                >
                    {#each availableWidthZones as each_zone}
                        <option>{each_zone}</option>
                    {/each}
                </select>
            </div>
            <div>Height zones</div>
            <div>
                <select
                    on:change={onZoneNumberChange}
                    bind:value={nHeightZones}
                    class="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-md block px-4 py-1"
                >
                    {#each availableHeightZones as each_zone}
                        <option>{each_zone}</option>
                    {/each}
                </select>
            </div>
        </controls> -->

        <gradient-control class="block mt-4">
            <input
                checked
                id="checked-checkbox"
                type="checkbox"
                bind:value={useGradientPreview}
                class="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded"
            />
            <label for="checked-checkbox" class=""
                >Enable gradient in preview</label
            >
        </gradient-control>

        {#if !useGradientPreview}
            <preview class="block p-4">
                <div class="flex mx-[50px] w-[384px]">
                    {#each { length: nWidthZones } as _, i}
                        <width-zone style={colourList[0][i]} />
                    {/each}
                </div>
                <div class="flex justify-between">
                    <div class="flex flex-col h-[218px]">
                        {#each { length: nHeightZones } as _, i}
                            <height-zone style={colourList[2][i]} />
                        {/each}
                    </div>
                    <div class="flex flex-col h-[218px]">
                        {#each { length: nHeightZones } as _, i}
                            <height-zone style={colourList[3][i]} />
                        {/each}
                    </div>
                </div>
                <div class="flex mx-[50px] w-[384px]">
                    {#each { length: nWidthZones } as _, i}
                        <width-zone style={colourList[1][i]} />
                    {/each}
                </div>
            </preview>
        {/if}
        {#if useGradientPreview}
            <gradient-preview class="block" >
                <div class="w-[382px] h-[50px] mx-[50px]" style={gradientList[0]} />
                <div class="flex justify-between">
                    <div class="h-[218px] w-[50px]" style={gradientList[2]} />
                    <div class="h-[218px] w-[50px]" style={gradientList[3]} />
                </div>
                <div
                    class="w-[382px] h-[50px] mx-[50px]"
                    style={gradientList[1]}
                />
            </gradient-preview>
        {/if}
    {/if}
</main>

<style>
    width-zone {
        @apply w-full h-[50px];
    }

    height-zone {
        @apply w-[50px] h-full;
    }

    .grid-lock {
        background-image: linear-gradient(to right, rgb(20, 12, 31), rgb(132, 212, 112));
    }
</style>
